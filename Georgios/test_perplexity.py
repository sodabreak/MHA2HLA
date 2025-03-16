"""
Dependencies:
    torch,
    transformers (https://pypi.org/project/transformers/)
    lm-eval (https://pypi.org/project/lm-eval/)
    jsonargparse (https://pypi.org/project/jsonargparse/)

Small model checkpoints for dev:
    AICrossSim/clm-60m (https://huggingface.co/AICrossSim/clm-60m)
    AICrossSim/clm-200m (https://huggingface.co/AICrossSim/clm-200m)
    TinyLlama/TinyLlama_v1.1 (https://huggingface.co/TinyLlama/TinyLlama_v1.1)

Reference evaluation results:
    AICrossSim/clm-60m, wikitext
    $ python demo.py eval --model_name AICrossSim/clm-60m
    | Tasks  |Version|Filter|n-shot|    Metric     |   | Value  |   |Stderr|
    |--------|------:|------|-----:|---------------|---|-------:|---|------|
    |wikitext|      2|none  |     0|bits_per_byte  |↓  |  1.6732|±  |   N/A|
    |        |       |none  |     0|byte_perplexity|↓  |  3.1893|±  |   N/A|
    |        |       |none  |     0|word_perplexity|↓  |493.7035|±  |   N/A|

    AICrossSim/clm-200m, wikitext
    $ python demo.py eval --model_name AICrossSim/clm-200m
    | Tasks  |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
    |--------|------:|------|-----:|---------------|---|------:|---|------|
    |wikitext|      2|none  |     0|bits_per_byte  |↓  | 1.1067|±  |   N/A|
    |        |       |none  |     0|byte_perplexity|↓  | 2.1535|±  |   N/A|
    |        |       |none  |     0|word_perplexity|↓  |60.4573|±  |   N/A|
"""

from pathlib import Path
from typing import Optional, Union, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from jsonargparse import CLI


def transform_model_to_MLA(model: torch.nn.Module, transform_config: dict[str, dict]):
    """transform model to MLA here."""
    print("Transforming model to MLA...")
    return model


def eval(
    model_name: str = "AICrossSim/clm-60m",
    transform_config: Optional[Path] = None,
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    tasks: Optional[list[str]] = ["wikitext"],
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = 4,
    max_seq_len: Optional[int] = 2048,
    limit: Optional[Union[int, float]] = None,
):
    """Evaluate a pretrained model using lm-eval.

    batch_size="auto" to use the maximum batch size that fits in GPU memory
    """
    device = torch.device("cuda")
    # *: create the model and transform it on CPU if needed
    # *: attn_implementation="eager" is required otherwise transformers will use FlashAttn or torch SDPA
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype), attn_implementation="eager"
    )
    model.eval()
    if transform_config is not None:
        # with open(transform_config, "r") as f:
        #     transform_config = yaml.safe_load(f)
        model = transform_model_to_MLA(model, transform_config)
    # # move the model to GPU
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # *: wrap the model with HFLM API
    model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_seq_len)

    results = simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )

    if results is not None:
        results.pop("samples")
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":

    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cli_map = {
        "eval": eval,
    }
    CLI(cli_map)
