# Method Naming Guide

This project separates three concepts:

- **Page framework**: how KV cache is split, protected, scored, and assigned
  precision.
- **Quant backend**: how one sealed page is encoded.
- **Experiment method**: a named combination used in PPL, NIAH, LongBench, and
  main comparison runs.

## Backend Names

| Backend | Meaning | SKVQ repo required |
|---|---|---:|
| `dynamic` | HuggingFace/default fp16 cache baseline. | No |
| `block_tq` | Page cache + TurboQuant backend, uniform K/V bits. | No |
| `block_tq_mix` | Page cache + TurboQuant backend + page-level mixed precision. | No |
| `block_skvq` | Page cache + in-repo SKVQ-style backend, uniform K/V bits. | No |
| `block_skvq_mix` | Page cache + in-repo SKVQ-style backend + page-level mixed precision. | No |
| `block_tq_pure` | Paper-aligned TurboQuant replacement path with SKVQ-style sink/window defaults. | No |
| `block_tq_pure_mix` | `block_tq_pure` plus page-level mixed precision and layer protection. | No |
| `v2_paper` | Legacy TurboQuant V2/QJL cache baseline used by PPL scripts. | No |
| `v3_flat` | TurboQuant V3 flat residual-window baseline, without page table. | No |
| `skvq_native` | External SKVQ paper implementation baseline. | Yes |

## Formal Main Experiment Methods

`experiment_main.py` uses `MethodSpec` objects from
`turboquant.block_cache.methods`.

| Method | Backend | Group | Purpose |
|---|---|---|---|
| `FP16` | `dynamic` | `reference` | Uncompressed reference. |
| `SKVQ Baseline` | `block_skvq` | `baseline` | In-framework SKVQ-style page compressor with token policy. |
| `TurboQuant Baseline` | `block_tq` | `baseline` | In-framework TurboQuant page compressor with token policy. |
| `SKVQ skvq_baseline (native)` | `skvq_native` | `paper_baseline` | External SKVQ paper code path. |
| `TurboQuant V3 flat (rw=128, K2/V2)` | `v3_flat` | `paper_baseline` | Non-page TurboQuant V3 residual-window baseline. |
| `TurboQuant pure (tq_replace)` | `block_tq` | `paper_baseline` | TurboQuant quantizer inside paper-style sink/window defaults. |
| `TurboQuant pure+PageMix` | `block_tq_pure_mix` | `paper_baseline` | Paper-style sink/window plus page mixed precision. |
| `Hybrid+SKVQ+Block` | `block_skvq` | `method` | Page framework + SKVQ-style backend. |
| `Hybrid+TQ+Block` | `block_tq` | `method` | Page framework + TurboQuant backend. |
| `Hybrid+TQ+Block+PageMix` | `block_tq_mix` | `method` | Main KVcatch target: page framework + TurboQuant + mixed precision. |
| `Hybrid+TQ+RandomMix` | `block_tq_random_mix` | `ablation` | Random page-importance ablation when `--include-random-mix` is set. |

## Where Methods Are Defined

- `turboquant/block_cache/backends/`: compression algorithms for one page.
- `turboquant/block_cache/methods.py`: backend selection, `MethodSpec`, cache
  factories, and result-table method configs.
- `turboquant/block_cache/experiment_main.py`: orchestration of the formal
  comparison run.
- `turboquant/block_cache/eval_*.py`: task-specific evaluation loops.

## Adding A New Method

1. Add a backend under `turboquant/block_cache/backends/` if the method needs a
   new quantization algorithm.
2. Register it with `register_page_backend()`.
3. Add a backend name or `MethodSpec` in `methods.py`.
4. Reuse the existing eval scripts. They should not need method-specific logic
   unless the task itself changes.
