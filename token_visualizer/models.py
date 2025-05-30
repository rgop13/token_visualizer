#!/usr/bin/env python3

import json
import math
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import requests
import torch
from loguru import logger
from openai import OpenAI
from requests.adapters import HTTPAdapter
from retrying import retry
from urllib3.util import Retry
from torch.nn import functional as F

from .token_html import Token, tokens_info_to_html

__all__ = [
    "TopkTokenModel",
    "TransformerModel",
    "SentenceTransformerModel",
    "TGIModel",
    "OpenAIModel",
    "OpenAIProxyModel",
    "generate_topk_token_prob",
    "generate_topk_token_prob_sentence_transformer",
    "load_model_tokenizer",
    "openai_payload",
]


def load_model_tokenizer(repo):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def load_sentence_transformer_model_tokenizer(repo):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(repo, trust_remote_code=True)
    tokenizer = model.tokenizer
    return model, tokenizer


def format_reverse_vocab(tokenizer) -> Dict[int, str]:
    """
    Format the vocab to make it more human-readable, return a token_id to token_value mapping.
    """
    rev_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    sp_space = b"\xe2\x96\x81".decode()  # reference link below in sentencepiece:
    # https://github.com/google/sentencepiece/blob/8cbdf13794284c30877936f91c6f31e2c1d5aef7/src/sentencepiece_processor.cc#L41-L42

    for idx, token in rev_vocab.items():
        if sp_space in token:
            rev_vocab[idx] = token.replace(sp_space, "␣")
        if token.startswith('##'):
            rev_vocab[idx] = token.replace('##', '')
        elif token.isspace():  # token like \n, \t or multiple spaces
            rev_vocab[idx] = repr(token)[1:-1]  # 1:-1 to strip ', it will convert \n to \\n
        elif token.startswith("<") and token.endswith(">"):  # tokens like <s>
            # NOTE: string like <pre>&lt;s&gt;</pre> is better, but <|s|> is simple, better-looking
            # rev_vocab[idx] = f"<pre>&lt;{token[1:-1]}&gt;</pre>"
            rev_vocab[idx] = f"<|{token[1:-1]}|>"

    return rev_vocab


def generate_topk_token_prob(
    inputs: str, model, tokenizer,
    num_topk_tokens: int = 10,
    inputs_device: str = "cuda:0",
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate topk token and it's prob for each token of auto regressive model.
    """
    if not torch.cuda.is_available():
        inputs_device = "cpu"
        model = model.to(inputs_device)
        logger.warning(f"CUDA not available, switch to {inputs_device}.")

    logger.info(f"generate response for:\n{inputs}")
    inputs = tokenizer(inputs, return_tensors='pt')
    inputs = inputs.to(inputs_device)
    outputs = model.generate(
        **inputs,
        return_dict_in_generate=True,
        output_scores=True,
        **kwargs
    )
    logits = torch.stack(outputs.scores)
    probs = torch.softmax(logits, dim=-1)
    topk_tokens = torch.topk(logits, k=num_topk_tokens).indices
    topk_probs = torch.gather(probs, -1, topk_tokens)
    return topk_tokens, topk_probs, outputs.sequences


def generate_topk_token_prob_sentence_transformer(
    query: str, document: str, model, tokenizer,
    num_topk_tokens: int = 10,
    inputs_device: str = "cuda:0",
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """
    Generate topk token and it's prob for each token of auto regressive model.
    """
    if not torch.cuda.is_available():
        inputs_device = "cpu"
        model = model.to(inputs_device)
        logger.warning(f"CUDA not available, switch to {inputs_device}.")

    logger.info(f"generate response for:\n{query}\n{document}")
    query_embeddings = model.encode(
        query, 
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    query_token_embedidngs = model.encode(
        query, 
        convert_to_tensor=True,
        show_progress_bar=False,
        output_value="token_embeddings",
        normalize_embeddings=True,
    )
    pos_embeddings = model.encode(
        document, 
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    pos_token_embeddings = model.encode(
        document, 
        convert_to_tensor=True,
        show_progress_bar=False,
        output_value="token_embeddings",
        normalize_embeddings=True,
    )
    
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    positive_ids = tokenizer.encode(document, add_special_tokens=False)
    
    qp_elementwise_product = query_embeddings * pos_embeddings
    # qp_tokenwise_similarity = model.similarity(query_token_embedidngs[1:-1], pos_token_embeddings[1:-1]).squeeze()   # (num_q_tokens, num_p_tokens)
    # q_token_probs = torch.softmax(qp_tokenwise_similarity, dim=0)   # (num_q_tokens)
    # p_token_probs = torch.softmax(qp_tokenwise_similarity, dim=1)   # (num_p_tokens)
    q_token_contributions = model.similarity(query_token_embedidngs, qp_elementwise_product).squeeze()[1:-1]    # (num_q_tokens)
    p_token_contributions = model.similarity(pos_token_embeddings, qp_elementwise_product).squeeze()[1:-1]    # (num_p_tokens)
    
    return q_token_contributions, p_token_contributions, query_ids, positive_ids


def openai_top_response_tokens(response: Dict) -> List[Token]:
    token_logprobs = response["choices"][0]["logprobs"]["content"]
    tokens = []
    for token_prob in token_logprobs:
        prob = math.exp(token_prob["logprob"])
        candidate_tokens = [
            Token(t["token"], math.exp(t["logprob"]))
            for t in token_prob["top_logprobs"]
        ]
        token = Token(token_prob["token"], prob, top_candidates=candidate_tokens)
        tokens.append(token)
    return tokens


def openai_payload(
    prompt: Union[List[str], str],
    model_name: str,
    system_prompt: str = "",
    **kwargs
) -> Dict:
    """Generate payload for openai api call."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if isinstance(prompt, str):
        prompt = [prompt]
    for idx, p in enumerate(prompt):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append({"role": role, "content": p})

    payload = {"model": model_name, "messages": messages, **kwargs}
    return payload


@dataclass
class TopkTokenModel:
    do_sample: bool = False
    temperature: float = 1.0
    max_tokens: int = 4096
    repetition_penalty: float = 1.0
    num_beams: int = 1
    topk: int = 50
    topp: float = 1.0

    topk_per_token: int = 5  # number of topk tokens to generate for each token
    generated_answer: Optional[str] = None  # generated answer from model, to display in frontend
    display_whitespace: bool = False

    def generate_topk_per_token(self, text: str) -> List[Token]:
        """
        Generate prob, text and candidates for each token of the model's output.
        This function is used to visualize the inference process.
        """
        raise NotImplementedError

    def generate_inputs_prob(self, text: str) -> List[Token]:
        """
        Generate prob and text for each token of the input text.
        This function is used to visualize the ppl.
        """
        raise NotImplementedError

    def html_to_visualize(self, tokens: List[Token], space: List[bool]) -> str:
        """Generate html to visualize the tokens."""
        return tokens_info_to_html(tokens, display_whitespace=self.display_whitespace, space=space)


@dataclass
class TransformerModel(TopkTokenModel):

    repo: Optional[str] = None
    model = None
    tokenizer = None
    rev_vocab = None

    def get_model_tokenizer(self):
        assert self.repo, "Please provide repo name to load model and tokenizer."
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_model_tokenizer(self.repo)
        if self.rev_vocab is None:
            self.rev_vocab = format_reverse_vocab(self.tokenizer)
        return self.model, self.tokenizer

    def generate_topk_per_token(self, text: str) -> List[Token]:
        model, tokenizer = self.get_model_tokenizer()
        rev_vocab = self.rev_vocab
        assert rev_vocab, f"Reverse vocab not loaded for {self.repo} model"

        topk_tokens, topk_probs, sequences = generate_topk_token_prob(
            text, model, tokenizer, num_topk_tokens=self.topk_per_token,
            do_sample=self.do_sample,
            temperature=max(self.temperature, 0.01),
            max_new_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            num_beams=self.num_beams,
            top_k=self.topk,
            top_p=self.topp,
        )
        self.generated_answer = tokenizer.decode(sequences[0])

        seq_length = topk_tokens.shape[0]
        np_seq = sequences[0, -seq_length:].cpu().numpy()
        gen_tokens = []
        for seq_id, token, prob in zip(np_seq, topk_tokens.cpu().numpy(), topk_probs.cpu().numpy()):
            candidate_tokens = [Token(f"{rev_vocab[idx]}", float(p)) for idx, p in zip(token[0], prob[0])]   # noqa
            seq_id_prob = float(prob[0][token[0] == seq_id])
            display_token = Token(f"{rev_vocab[seq_id]}", seq_id_prob, candidate_tokens)
            gen_tokens.append(display_token)
        return gen_tokens
    

@dataclass
class SentenceTransformerModel(TopkTokenModel):

    repo: Optional[str] = None
    model = None
    tokenizer = None
    rev_vocab = None

    def get_model_tokenizer(self):
        assert self.repo, "Please provide repo name to load model and tokenizer."
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_sentence_transformer_model_tokenizer(self.repo)
        if self.rev_vocab is None:
            self.rev_vocab = format_reverse_vocab(self.tokenizer)
        return self.model, self.tokenizer

    def generate_topk_per_token(self, query: str, document: str) -> Tuple[List[Token], List[Token]]:
        model, tokenizer = self.get_model_tokenizer()
        rev_vocab = self.rev_vocab
        assert rev_vocab, f"Reverse vocab not loaded for {self.repo} model"

        q_token_contributions, p_token_contributions, query_ids, positive_ids = generate_topk_token_prob_sentence_transformer(
            query, document, model, tokenizer, num_topk_tokens=self.topk_per_token,
            do_sample=self.do_sample,
            temperature=max(self.temperature, 0.01),
            max_new_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            num_beams=self.num_beams,
            top_k=self.topk,
            top_p=self.topp,
        )
        query_space = []
        for word in query.split():
            tokens = tokenizer.tokenize(word)
            spaces = [False] * (len(tokens) - 1) + [True]
            query_space.extend(spaces)
        # query_space.append(True)
        pos_space = []
        for word in document.split():
            tokens = tokenizer.tokenize(word)
            spaces = [False] * (len(tokens) - 1) + [True]
            pos_space.extend(spaces)
        # pos_space.append(True)
        self.query_tokens = tokenizer.decode(query_ids)
        self.query_space = query_space
        self.positive_tokens = tokenizer.decode(positive_ids)
        self.positive_space = pos_space
        seq_length = q_token_contributions.shape[0]
        np_seq = np.array(query_ids)[-seq_length:]
        query_tokens_display = []
        for seq_id, token, prob in zip(np_seq, query_ids, q_token_contributions.cpu().numpy()):
            display_token = Token(f"{rev_vocab[seq_id]}", prob)
            query_tokens_display.append(display_token)
            
        seq_length = p_token_contributions.shape[0]
        np_seq = np.array(positive_ids)[-seq_length:]
        positive_tokens_display = []
        for seq_id, token, prob in zip(np_seq, positive_ids, p_token_contributions.cpu().numpy()):
            display_token = Token(f"{rev_vocab[seq_id]}", prob)
            positive_tokens_display.append(display_token)
        return query_tokens_display, positive_tokens_display


def tgi_response(  # type: ignore[return]
    input_text: str,
    url: str,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.1,
    temperature: float = 0.01,
    top_k: int = 5,
    top_p: float = 0.85,
    do_sample: bool = True,
    topk_logits: Optional[int] = None,
    details: bool = False,
    **kwargs
) -> Dict:
    headers = {"Content-Type": "application/json"}
    params = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_n_tokens": topk_logits,
        "details": details,
        **kwargs,
    }
    if do_sample:  # tgi use or logic for top_k/top_p with do_sample
        params.update({"top_k": top_k, "top_p": top_p})

    data = {"inputs": input_text, "parameters": params}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        logger.error(f"Error {response.status_code}: {response.text}")
    return response.json()


@dataclass
class TGIModel(TopkTokenModel):

    url: Optional[str] = None
    system_prompt = ""
    details: bool = False
    decoder_input_details: bool = False  # input logprobs
    num_prefill_tokens: Optional[int] = None

    # tgi support top_n_tokens, reference below:
    # https://github.com/huggingface/text-generation-inference/blob/7dbaf9e9013060af52024ea1a8b361b107b50a69/proto/generate.proto#L108-L109

    def response_to_inputs(self, inputs: str) -> Dict:
        assert self.url, f"Please provide url to access tgi api. url: {self.url}"
        json_response = tgi_response(
            inputs, url=self.url,
            max_new_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_k=self.topk,
            top_p=min(self.topp, 0.99),
            do_sample=self.do_sample,
            decoder_input_details=self.decoder_input_details,
            topk_logits=self.topk_per_token,
            details=self.details,
        )
        response = json_response[0]
        self.generated_answer = response["generated_text"]
        return response

    def generate_topk_per_token(self, text: str) -> List[Token]:
        assert self.details, "Please set details to True."
        response = self.response_to_inputs(text)
        tokens: List[Token] = []

        token_details = response["details"]["tokens"]
        topk_tokens = response["details"]["top_tokens"]

        for details, candidate in zip(token_details, topk_tokens):
            candidate_tokens = [Token(x["text"], math.exp(x["logprob"])) for x in candidate]
            token = Token(
                details["text"],
                math.exp(details["logprob"]),
                top_candidates=candidate_tokens,
            )
            tokens.append(token)

        return tokens

    def generate_inputs_prob(self, text: str) -> List[Token]:
        assert self.decoder_input_details, "Please set decoder_input_details to True."
        response = self.response_to_inputs(text)
        token_details = response["details"]["prefill"]
        tokens = []
        for token in token_details:
            logprob = token.get("logprob", None)
            if logprob is None:
                continue
            tokens.append(Token(token["text"], math.exp(logprob)))
        return tokens

    def set_num_prefill_tokens(self, response):
        if self.details:
            self.num_prefill_tokens = response["details"]["prefill_tokens"]
        else:
            self.num_prefill_tokens = None


@dataclass
class OpenAIModel(TopkTokenModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    system_prompt: str = ""
    model_name: str = "gpt-4-0125-preview"
    # choices for model_name: see https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    json_mode: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        assert self.api_key is not None, "Please provide api key to access openai api."
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_topk_per_token(self, text: str, **kwargs) -> List[Token]:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.topp,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.topk_per_token > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.topk_per_token

        payload = openai_payload(text, self.model_name, system_prompt=self.system_prompt, **kwargs)
        completion = self.client.completions.create(payload)
        self.generated_answer = completion.choices[0].message.content
        return openai_top_response_tokens(completion.dict())


@dataclass
class OpenAIProxyModel(TopkTokenModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    system_prompt = ""
    model_name: str = "gpt-4-0125-preview"
    # choices for model_name: see https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    json_mode: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        assert self.base_url is not None, "Please provide url to access openai api."
        assert self.api_key is not None, "Please provide api key to access openai api."
        retry_strategy = Retry(
            total=1,  # max retry times
            backoff_factor=1,  # time interval between retries
            status_forcelist=[429, 500, 502, 503, 504],  # retry when these status code
            allowed_methods=["POST"],  # retry only when POST
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    @retry(stop_max_attempt_number=3)
    def openai_api_call(self, payload):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        response = self.session.post(self.base_url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            err_msg = f"Access openai error, status code: {response.status_code}, errmsg: {response.text}"   # noqa
            raise ValueError(err_msg, response.status_code)

        data = json.loads(response.text)
        return data

    def generate_topk_per_token(self, text: str, **kwargs) -> List[Token]:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.topp,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.topk_per_token > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.topk_per_token

        payload = openai_payload(text, self.model_name, system_prompt=self.system_prompt, **kwargs)
        response = self.openai_api_call(payload)
        self.generated_answer = response["choices"][0]["message"]["content"]
        return openai_top_response_tokens(response)


if __name__ == "__main__":
    model = SentenceTransformerModel(repo="sentence-transformers/all-MiniLM-L6-v2")
    query = "What is the capital of France?"
    document = "France is a country in Western Europe."
    query_tokens, positive_tokens = model.generate_topk_per_token(query, document)
    print(query_tokens)
    print(positive_tokens)
