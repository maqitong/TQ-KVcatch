"""Side-by-side benchmark: FP16 baseline vs three grouping policies.

Reports compression ratio, generated text, and (where possible) the
needle-in-haystack accuracy used elsewhere in this repo.

Run example:
    python -m turboquant.block_cache.benchmark --model Qwen/Qwen3-4B-Instruct
    python -m turboquant.block_cache.benchmark \
        --model meta-llama/Meta-Llama-3-8B-Instruct --load-in-8bit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from turboquant.block_cache import (
    BlockCacheConfig,
    BlockKVCache,
    HybridPolicy,
    TokenBlockPolicy,
    WindowBlockPolicy,
)


SECRET = "AURORA-7749"
HAYSTACK_TEMPLATE = (
    "The following is a long context document. {filler}"
    "By the way, the secret project code name is {secret}. {filler}"
    "End of document.\n\n"
    "Question: what is the secret project code name?\nAnswer:"
)


@dataclass
class RunResult:
    name: str
    text: str
    seconds: float
    ratio: float
    contains_secret: bool
    cache_report: dict | None


def make_haystack(approx_tokens: int, tokenizer) -> str:
    filler = ("This is irrelevant filler content. " * 4 + "\n") * 16
    seed = HAYSTACK_TEMPLATE.format(filler=filler, secret=SECRET)
    while len(tokenizer(seed).input_ids) < approx_tokens:
        seed = seed.replace("End of document.", filler + "End of document.")
    return seed


def run_once(
    name: str,
    model,
    tokenizer,
    prompt: str,
    cache: BlockKVCache | None,
    max_new_tokens: int,
) -> RunResult:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if cache is not None:
        cache.reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    dt = time.perf_counter() - t0
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    report = cache.memory_report() if cache is not None else None
    ratio = report["compression_ratio"] if report else 1.0
    return RunResult(
        name=name,
        text=text.strip(),
        seconds=dt,
        ratio=ratio,
        contains_secret=SECRET in text,
        cache_report=report,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--ctx-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--key-bits", type=int, default=6)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kw = {"torch_dtype": torch.float16}
    if args.load_in_8bit:
        kw.pop("torch_dtype", None)
        kw["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", **kw
    )
    model.eval()

    prompt = make_haystack(args.ctx_tokens, tokenizer)
    n_in = len(tokenizer(prompt).input_ids)
    print(f"Prompt length: ~{n_in} tokens")

    configs = [
        ("FP16 baseline", None),
        (
            f"TokenBlock K{args.key_bits}V{args.value_bits} bs={args.block_size}",
            BlockCacheConfig(
                block_size=args.block_size,
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                policy=TokenBlockPolicy(),
            ),
        ),
        (
            f"WindowBlock K{args.key_bits}V{args.value_bits} win={args.window}",
            BlockCacheConfig(
                block_size=args.block_size,
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                policy=WindowBlockPolicy(window_size=args.window),
            ),
        ),
        (
            f"Hybrid sink={args.sink} win={args.window} K{args.key_bits}V{args.value_bits}",
            BlockCacheConfig(
                block_size=args.block_size,
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                policy=HybridPolicy(sink_size=args.sink, window_size=args.window),
            ),
        ),
    ]

    results: list[RunResult] = []
    for name, cfg in configs:
        cache = BlockKVCache(cfg) if cfg is not None else None
        print(f"\n--- running: {name} ---")
        try:
            r = run_once(name, model, tokenizer, prompt, cache, args.max_new_tokens)
        except Exception as e:
            print(f"  ERROR: {e!r}")
            continue
        results.append(r)
        print(f"  time:  {r.seconds:.2f}s")
        print(f"  ratio: {r.ratio:.2f}x")
        print(f"  found secret: {r.contains_secret}")
        print(f"  output: {r.text[:200]!r}")

    print("\n=== summary ===")
    print(f"{'config':<60} {'ratio':>8} {'sec':>8} {'found':>7}")
    for r in results:
        print(
            f"{r.name:<60} {r.ratio:>7.2f}x {r.seconds:>7.2f}s "
            f"{('YES' if r.contains_secret else '-'):>7}"
        )


if __name__ == "__main__":
    main()
