"""Minimal demo: HuggingFace generate() with `BlockKVCache`.

Default model is Qwen3-4B-Instruct; pass `--model` for another HF id and
`--policy` to switch between token / window / hybrid grouping.

Run examples:
    python -m turboquant.block_cache.demo
    python -m turboquant.block_cache.demo --policy window --window 128
    python -m turboquant.block_cache.demo --model meta-llama/Meta-Llama-3-8B-Instruct \
        --load-in-8bit --policy hybrid --sink 4 --window 64
"""

from __future__ import annotations

import argparse
import textwrap

import torch

from turboquant.block_cache import (
    BlockCacheConfig,
    BlockKVCache,
    HybridPolicy,
    TokenBlockPolicy,
    WindowBlockPolicy,
)


PROMPT = textwrap.dedent("""\
    You are a careful technical writer. Briefly explain how a key-value cache
    in a transformer decoder grows during autoregressive generation, and why
    quantizing older tokens more aggressively than recent ones can be a good
    memory tradeoff. Keep the answer to four sentences.
""")


def build_policy(args):
    if args.policy == "token":
        return TokenBlockPolicy()
    if args.policy == "window":
        return WindowBlockPolicy(window_size=args.window)
    if args.policy == "hybrid":
        return HybridPolicy(sink_size=args.sink, window_size=args.window)
    raise ValueError(args.policy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument(
        "--policy", choices=["token", "window", "hybrid"], default="window"
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--key-bits", type=int, default=6)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument(
        "--granularity", choices=["per-vector", "per-block"], default="per-vector"
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {"torch_dtype": torch.float16}
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs.pop("torch_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", **model_kwargs
    )
    model.eval()

    cfg = BlockCacheConfig(
        block_size=args.block_size,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        granularity=args.granularity,
        policy=build_policy(args),
    )
    cache = BlockKVCache(cfg)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n=== generated ===\n" + text)

    report = cache.memory_report()
    print("\n=== memory report ===")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
