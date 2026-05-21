# KVcatch Main Experiment

## NIAH Summary

| Method | Group | Policy | Scheme | Context | Found | Total | Found Rate | Avg Ratio | Avg Seconds |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | reference | none | FP16 | 2048 | 9 | 9 | 1.000 | - | 0.467 |
| SKVQ Baseline | baseline | token | Uniform K2/V2 | 2048 | 0 | 9 | 0.000 | 5.786 | 106.603 |
| TurboQuant Baseline | baseline | token | Uniform K2/V2 | 2048 | 3 | 9 | 0.333 | 6.271 | 13.290 |
| SKVQ skvq_baseline (native) | paper_baseline | sliding_window | Uniform K2/V2 | 2048 | 4 | 9 | 0.444 | - | 1.351 |
| TurboQuant V3 flat (rw=128, K2/V2) | paper_baseline | v3_flat | V3 flat K2/V2 | 2048 | 1 | 9 | 0.111 | 7.529 | 2.178 |
| TurboQuant pure (tq_replace) | paper_baseline | hybrid | Uniform K2/V2 | 2048 | 7 | 9 | 0.778 | 5.155 | 19.548 |
| TurboQuant pure+PageMix | paper_baseline | hybrid | Mixed high K4/V4, low K2/V2 | 2048 | 8 | 9 | 0.889 | 4.015 | 28.569 |
| Hybrid+SKVQ+Block | method | hybrid | Uniform K2/V2 | 2048 | 3 | 9 | 0.333 | 4.292 | 98.555 |
| Hybrid+TQ+Block | method | hybrid | Uniform K2/V2 | 2048 | 2 | 9 | 0.222 | 4.500 | 12.993 |
| Hybrid+TQ+Block+PageMix | method | hybrid | Mixed high K4/V4, low K2/V2 | 2048 | 2 | 9 | 0.222 | 3.914 | 21.894 |
| Hybrid+TQ+RandomMix | ablation | hybrid | Mixed high K4/V4, low K2/V2 | 2048 | 5 | 9 | 0.556 | 3.911 | 21.771 |
| FP16 | reference | none | FP16 | 4096 | 9 | 9 | 1.000 | - | 0.702 |
| SKVQ Baseline | baseline | token | Uniform K2/V2 | 4096 | 0 | 9 | 0.000 | 5.784 | 2226.382 |
| TurboQuant Baseline | baseline | token | Uniform K2/V2 | 4096 | 1 | 9 | 0.111 | 6.358 | 119.848 |
| SKVQ skvq_baseline (native) | paper_baseline | sliding_window | Uniform K2/V2 | 4096 | 0 | 9 | 0.000 | - | 2.547 |
| TurboQuant V3 flat (rw=128, K2/V2) | paper_baseline | v3_flat | V3 flat K2/V2 | 4096 | 3 | 9 | 0.333 | 7.529 | 2.956 |
| TurboQuant pure (tq_replace) | paper_baseline | hybrid | Uniform K2/V2 | 4096 | 6 | 9 | 0.667 | 6.191 | 31.368 |
| TurboQuant pure+PageMix | paper_baseline | hybrid | Mixed high K4/V4, low K2/V2 | 4096 | 6 | 9 | 0.667 | 4.590 | 40.639 |
| Hybrid+SKVQ+Block | method | hybrid | Uniform K2/V2 | 4096 | 0 | 9 | 0.000 | 4.948 | 2145.692 |
| Hybrid+TQ+Block | method | hybrid | Uniform K2/V2 | 4096 | 2 | 9 | 0.222 | 5.308 | 55.299 |
| Hybrid+TQ+Block+PageMix | method | hybrid | Mixed high K4/V4, low K2/V2 | 4096 | 1 | 9 | 0.111 | 4.504 | 66.197 |
| Hybrid+TQ+RandomMix | ablation | hybrid | Mixed high K4/V4, low K2/V2 | 4096 | 6 | 9 | 0.667 | 4.517 | 61.161 |

## Settings

```json
{
  "model": "/root/autodl-tmp/bt-kvcatch/models",
  "context_lengths": [
    2048,
    4096
  ],
  "positions": [
    0.1,
    0.5,
    0.9
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "block_size": 16,
  "sink": 5,
  "window": 128,
  "key_bits": 2,
  "value_bits": 2,
  "group_size": 128,
  "key_group_size": null,
  "value_group_size": null,
  "max_cached_decompressed_blocks": 0,
  "important_ratio": 0.3,
  "high_bits": [
    4,
    4
  ],
  "low_bits": [
    2,
    2
  ],
  "num_layers": 32,
  "protected_layers": 1,
  "protected_bits": [
    8,
    8
  ]
}
```
