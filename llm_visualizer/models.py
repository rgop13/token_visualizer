#!/usr/bin/env python3

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import requests
import torch
from openai import OpenAI
from loguru import logger
from requests.adapters import HTTPAdapter
from retrying import retry
from urllib3.util import Retry

from .token_html import Token, tokens_info_to_html

__all__ = [
    "TopkTokenModel",
    "TransformerModel",
    "TGIModel",
    "OpenAIModel",
    "OpenAIProxyModel",
    "generate_topk_token_prob",
    "load_model_tokenizer",
    "openai_payload",
]


def load_model_tokenizer(repo):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True, trust_remote_code=True)
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
    generated_answer: str = None  # generated answer from model, to display in frontend

    def genearte_topk_per_token(self, text: str) -> List[Token]:
        raise NotImplementedError

    def html_to_visualize(self, tokens: List[Token]) -> str:
        return tokens_info_to_html(tokens, special_for_newline=False)


@dataclass
class TransformerModel(TopkTokenModel):

    repo: str = None
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

    def genearte_topk_per_token(self, text: str) -> List[Token]:
        model, tokenizer = self.get_model_tokenizer()
        rev_vocab = self.rev_vocab
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
class TGIModel(TopkTokenModel):
    # tgi support top_n_tokens, reference below:
    # https://github.com/huggingface/text-generation-inference/blob/7dbaf9e9013060af52024ea1a8b361b107b50a69/proto/generate.proto#L108-L109

    def genearte_topk_per_token(self, text: str) -> List[Token]:
        raise NotImplementedError


@dataclass
class OpenAIModel(TopkTokenModel):
    api_key: str = None
    base_url = None

    system_prompt: str = ""
    model_name: str = "gpt-4-0125-preview"
    # choices for model_name: see https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    json_mode: bool = False
    logprobs: bool = False
    seed: int = None
    top_logprobs: int = 5

    def __post_init__(self):
        assert self.api_key is not None, "Please provide api key to access openai api."
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def genearte_topk_per_token(self, text: str, **kwargs) -> List[Token]:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.topp,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.top_logprobs > 0:
            kwargs["logprobs"] = True
            assert self.top_logprobs >= 1 and self.top_logprobs <= 5
            kwargs["top_logprobs"] = self.top_logprobs

        payload = openai_payload(text, self.model_name, system_prompt=self.system_prompt, **kwargs)
        completion = self.client.completions.create(payload)
        self.generated_answer = completion.choices[0].message.content
        return openai_top_response_tokens(completion.dict())


@dataclass
class OpenAIProxyModel(TopkTokenModel):
    api_key: str = None
    base_url: str = None

    system_prompt = ""
    model_name: str = "gpt-4-0125-preview"
    # choices for model_name: see https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    json_mode: bool = False
    logprobs: bool = False
    seed: int = None
    top_logprobs: int = 5

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
            err_msg = f"Access openai error, status code: {response.status_code}, errmsg: {response.text}"
            raise ValueError(err_msg, response.status_code)

        data = json.loads(response.text)
        return data

    def genearte_topk_per_token(self, text: str, **kwargs) -> List[Token]:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.topp,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.top_logprobs > 0:
            kwargs["logprobs"] = True
            assert self.top_logprobs >= 1 and self.top_logprobs <= 5
            kwargs["top_logprobs"] = self.top_logprobs

        payload = openai_payload(text, self.model_name, system_prompt=self.system_prompt, **kwargs)
        response = self.openai_api_call(payload)
        self.generated_answer = response["choices"][0]["message"]["content"]
        return openai_top_response_tokens(response)
