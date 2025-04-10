#!/usr/bin/env python3
import os
import json
import jsonlines
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import functools
from argparse import ArgumentParser
from typing import Tuple, Optional

import gradio as gr
from loguru import logger

import token_visualizer
from token_visualizer import TopkTokenModel, css_style, ensure_os_env


def make_parser() -> ArgumentParser:
    parser = ArgumentParser("Inference process visualizer")
    parser.add_argument(
        "-t", "--type",
        choices=["llm", "tgi", "oai", "oai-proxy"],
        default="oai-proxy",
        help="Type of model to use, default to openai-proxy"
    )
    parser.add_argument(
        "--hf-repo", type=str, default="intfloat/e5-large-v2",  # intfloat/e5-large-v2
        help="Huggingface model repository, used when type is 'llm'. Default to None"
    )
    parser.add_argument(
        "--oai-model", type=str, default="gpt-4-turbo-2024-04-09",
        help="OpenAI model name, used when type is 'oai'/'oai-proxy'. "
        "Check https://platform.openai.com/docs/models for more details. "
        "Default to `gpt-4-turbo-2024-04-09`."
    )
    parser.add_argument(
        "--oai-key", type=str, default=None,
        help="OpenAI api key, used when type is 'oai'/'oai-proxy'. "
        "If provided, will override OPENAI_KEY env variable.",
    )
    parser.add_argument(
        "--tgi-url", type=str, default=None,
        help="Service url of TGI model, used when type is 'tgi'. "
        "If provided, will override TGI_URL env variable.",
    )
    parser.add_argument(
        "-s", "--share", action="store_true",
        help="Share service to the internet.",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=12123,
        help="Port to run the service, default to 12123."
    )
    return parser


def build_model_by_args(args) -> token_visualizer.TopkTokenModel:
    BASE_URL = ensure_os_env("BASE_URL")
    OPENAI_API_KEY = ensure_os_env("OPENAI_KEY")
    TGI_URL = ensure_os_env("TGI_URL")

    model: Optional[token_visualizer.TopkTokenModel] = None
    model = token_visualizer.SentenceTransformerModel(repo=args.hf_repo)

    # if args.type == "llm":
    #     model = token_visualizer.TransformerModel(repo=args.hf_repo)
    # elif args.type == "tgi":
    #     if args.tgi_url:
    #         TGI_URL = args.tgi_url
    #     model = token_visualizer.TGIModel(url=TGI_URL, details=True)
    # elif args.type == "oai":
    #     model = token_visualizer.OpenAIModel(
    #         base_url=BASE_URL,
    #         api_key=OPENAI_API_KEY,
    #         model_name=args.oai_model,
    #     )
    # elif args.type == "oai-proxy":
    #     model = token_visualizer.OpenAIProxyModel(
    #         base_url=BASE_URL,
    #         api_key=OPENAI_API_KEY,
    #         model_name="gpt-4-turbo-2024-04-09",
    #     )
    # else:
    #     raise ValueError(f"Unknown model type {args.type}")

    return model


@logger.catch(reraise=True)
def text_analysis(
    query: str,
    document: str,
    model: TopkTokenModel,  # model should be built in the interface
) -> Tuple[str, str, str]:

    query_tokens, positive_tokens = model.generate_topk_per_token(query, document)
    query_space = model.query_space
    positive_space = model.positive_space
    html_query = model.html_to_visualize(query_tokens, query_space)
    html_positive = model.html_to_visualize(positive_tokens, positive_space)

    html_query += "<br>"
    if isinstance(model, token_visualizer.TGIModel) and model.num_prefill_tokens:
        html_query += f"<div><strong>input tokens: {model.num_prefill_tokens}</strong></div>"
    html_query += f"<div><strong>query tokens: {len(query_tokens)}</strong></div>"
    html_positive += "<br>"
    if isinstance(model, token_visualizer.TGIModel) and model.num_prefill_tokens:
        html_positive += f"<div><strong>input tokens: {model.num_prefill_tokens}</strong></div>"
    html_positive += f"<div><strong>positive tokens: {len(positive_tokens)}</strong></div>"
    return model.query_tokens, html_query, model.positive_tokens, html_positive


def build_inference_analysis_demo(args):
    model = build_model_by_args(args)
    inference_func = functools.partial(text_analysis, model=model)
    
    base_dir = "/data/sjy/project/result/mining_result"
    data_name = "nq_dev_Llama-3.1-70B-Instruct_positive.jsonl"
    data_path = os.path.join(base_dir, data_name)
    examples = []
    with jsonlines.open(data_path) as reader:
        for line in reader:
            query = line["anchor"]
            positive = line["positive"]
            generated_positive = line["generated_positives"][0]
            generated_positive = generated_positive.split(':')[1:]
            generated_positive = ' '.join(generated_positive)
            examples.append([query, positive])
            examples.append([query, generated_positive])
            if len(examples) >= 200:
                break
    interface = gr.Interface(
        inference_func,
        inputs=[
            gr.TextArea(placeholder="Please input query here", label="Query"),
            gr.TextArea(placeholder="Please input document here", label="Document"),
        ],
        outputs=[
            gr.TextArea(label="Query tokens"),
            gr.HTML(label="Query tokens html"),
            gr.TextArea(label="Positive tokens"),
            gr.HTML(label="Positive tokens html"),
        ],
        examples=examples,
        title="Retrieval analysis",
    )
    return interface


@logger.catch(reraise=True)
def ppl_from_model(
    text: str,
    url: str,
    bos: str,
    eos: str,
    display_whitespace: bool,
    model,
) -> str:
    """Generate PPL visualization from model.

    Args:
        text (str): input text to visualize.
        url (str): tgi url.
        bos (str): begin of sentence token.
        eos (str): end of sentence token.
        display_whitespace (bool): whether to display whitespace for output text.
            If set to True, whitespace will be displayed as "␣".
    """
    url = url.strip()
    assert url, f"Please provide url of your tgi model. Current url: {url}"
    logger.info(f"Set url to {url}")
    model.url = url
    model.display_whitespace = display_whitespace
    model.max_tokens = 1

    text = bos + text + eos
    tokens = model.generate_inputs_prob(text)
    html = model.html_to_visualize(tokens)

    # display tokens and ppl at the end
    html += "<br>"
    html += f"<div><strong>total tokens: {len(tokens)}</strong></div>"
    ppl = tokens[-1].ppl
    html += f"<div><strong>ppl: {ppl:.4f}</strong></div>"
    return html


def build_ppl_visualizer_demo(args):
    model = build_model_by_args(args)
    ppl_func = functools.partial(ppl_from_model, model=model)

    ppl_interface = gr.Interface(
        ppl_func,
        [
            gr.TextArea(placeholder="Please input text to visualize here"),
            gr.TextArea(
                placeholder="Please input tgi url here (Error if not provided)",
                lines=1,
            ),
            gr.TextArea(placeholder="BOS token, default to empty string", lines=1),
            gr.TextArea(placeholder="EOS token, default to empty string", lines=1),
            gr.Checkbox(value=False, label="display whitespace in output, default to False"),
        ],
        "html",
        title="PPL Visualizer",
    )
    return ppl_interface


def demo():
    args = make_parser().parse_args()
    logger.info(f"Args: {args}")

    demo = gr.Blocks(css=css_style())
    with demo:
        with gr.Tab("Inference"):
            build_inference_analysis_demo(args)
        # with gr.Tab("PPL"):
            # build_ppl_visualizer_demo(args)

    demo.launch(
        server_name="0.0.0.0",
        share=args.share,
        server_port=args.port,
        
    )


if __name__ == "__main__":
    demo()
