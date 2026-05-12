# SKVQ + TurboQuant 集成 Todo

## 已读上下文

- [x] 阅读 `rules.md`：先写计划，动代码前需要确认。
- [x] 阅读 `SKVQ_plus_TurboQuant_方案.md`：目标是在 SKVQ 的 sink/window 调度里替换或叠加 TurboQuant 量化器，并与原 SKVQ 实验对比。
- [x] 阅读 `KV_process.py` / `KVcache_manager.py` / 评测脚本：SKVQ 的量化入口集中在 `SKVQuantProcessor.quantization()`，调度入口在 `ModelKVCacheManager.create()`，LongBench/Needle 各自有一份 `get_quantizer_from_str()`。
- [x] 阅读 `turboquant-pytorch` 的 `compressors_v3.py` / `lloyd_max.py` / `turboquant.py`：可复用的核心是 `generate_rotation_matrix()` 和 `LloydMaxCodebook`，先实现 fake-quant 的 dequantized 输出即可对齐 SKVQ 论文精度实验。

## 计划

- [x] 等待确认本计划。
- [x] 新增 `turboquant_backend.py`，实现一个 SKVQ 风格的 fake-quant 后端：
  - 支持 `k` / `v` 分别使用不同 bit。
  - 支持方案 A：按 head_dim 做 TurboQuant rotation + Lloyd-Max。
  - 支持方案 B：可选先走 SKVQ reorder，再在 group 内做 TurboQuant，最后逆 reorder。
  - 先只支持 fake quant；非 fake/pack 路径先显式报错，避免把存储格式和精度实验混在一起。
- [x] 小改 `KV_process.py`：
  - `SKVQuantProcessor` 增加可选 `tq_backend`。
  - 在 `quantization()` 中优先走 TurboQuant fake-quant 分支。
- [x] 小改 `KVcache_manager.py`：
  - `ModelKVCacheManager.create()` 增加 `turboquant_config`。
  - 每层创建对应 TurboQuant backend 并注入 processor。
  - 在 `tag()` 和打印信息里标出 `tq`，方便实验输出目录区分。
- [x] 抽出统一的量化字符串解析 helper，减少 `eval_longbench.py` 和 `eval_needle.py` 的重复逻辑，并给 `eval_ppl.py` 增加可选 `--quant` 单配置入口。
- [x] 扩展量化字符串，优先采用贴合现有解析顺序的格式：
  - 原 SKVQ：`k2-v2-g128-w128-reorder-pre_rope-clip-sink5-fp8`
  - 方案 A：`k4-v4-g128-w128-tq-sink5`
  - 方案 B：`k4-v4-g128-w128-tq-reorder-pre_rope-clip-sink5`
  - 方案 C：`k4-v2-g128-w128-tq-reorder-pre_rope-clip-sink5-protect4`
- [x] 增加轻量验证：
  - 构造小 tensor 验证 TurboQuant fake-quant 输出 shape/dtype/device 不变。
  - 验证 reorder + inverse reorder 路径能跑通。
  - 若本机依赖不足导致无法跑完整模型，只做 import/小 tensor 测试，并记录限制。
- [x] 更新 `SKVQ_plus_TurboQuant_方案.md` 中与实际实现不一致的部分，尤其是量化字符串格式、fake-quant 范围和分阶段实验路径。
- [x] 完成后在本文件增加 `Review` 部分，记录实际改动、验证结果和剩余风险。

## 当前实现取舍

- 第一轮只做精度对比所需的 fake quant，不做 bit-packed TurboQuant 存储格式。
- TurboQuant 只支持整数 bit；`k1.5-v1.5` 暂时保留给原 SKVQ，不纳入第一轮 TurboQuant 后端。
- 先不改 CUDA kernel；TurboQuant 路径走 PyTorch，便于快速验证实验趋势。

## Review

- 新增 `turboquant_backend.py`：实现 TurboQuant fake-quant 后端，包含 per-head 路径、reorder/group 路径、K/V 不同 bit、protected layers、norm clipping，以及缺少 `scipy` 时的内置 Gaussian Lloyd-Max fallback。
- 新增 `quant_parser.py`：统一解析 `k*-v*-g*-w*` 量化字符串，支持 `tq`、`protectN`、`sinkN`、`clip` / `clip0.xx`、`fp8`、`KIVI` 等关键字。
- 修改 `KV_process.py` / `KVcache_manager.py`：把 TurboQuant 作为可选后端注入，不影响原 SKVQ CUDA/PyTorch 分支。
- 修改 `eval_ppl.py` / `eval_longbench.py` / `eval_needle.py`：评测入口可以使用同一套 quant parser；PPL 额外支持 `--quant` 单配置实验。
- 更新 `D:\KVcatch\SKVQ_plus_TurboQuant_方案.md`：补充当前落地实现、支持范围和推荐实验字符串。
- 验证：`python -m py_compile turboquant_backend.py quant_parser.py KV_process.py KVcache_manager.py eval_ppl.py eval_longbench.py eval_needle.py` 通过；小 tensor 的 TurboQuant per-head 与 reorder/group fake-quant 路径均通过 shape/dtype/device 检查。
- 限制：当前环境没有安装 `skvq_quant` CUDA 扩展，因此未跑完整 PPL/LongBench/Needle；需要先按 README 执行 `cd kernels && pip install -e .`。

### Conda 环境

- 已创建 `skvq_tq`：Python 3.10，位置 `C:\Users\mqt\miniconda3\envs\skvq_tq`。
- 已安装 `torch==2.1.2+cu118`，在 NVIDIA T1000 8GB 上 `torch.cuda.is_available()` 为 `True`。
- 已安装 SKVQ 基础依赖，跳过 Windows 不可用/高风险的 `flash-attn` 和 pip 版 `triton`，并用 conda 版 `ninja` 替代 pip 版 `ninja`。
- 已调整 `KV_process.py`：缺少 `skvq_quant` CUDA 扩展时，fake-quant 自动退回 PyTorch 路径；packed CUDA dequant 仍要求扩展。
- 已调整 `eval_ppl.py` / `eval_longbench.py` / `eval_needle.py`：未安装 `flash-attn` 时不再强制启用 FlashAttention2。
- 环境验证通过：`pip check`、项目核心 import、评测脚本 `--help`、CUDA 上的 `SKVQuantProcessor + TurboQuantBackend` 小张量 smoke test。
- 已运行本地模型 `D:\model\Qwen2.5-3B-Instruct` 的 HuggingFace 生成 smoke test；升级 `transformers/tokenizers` 到 `4.43.1/0.19.1` 后可识别 `qwen2`，短 prompt 生成成功，峰值显存约 `6.446GB`。
- 剩余限制：`calib_config.py` 里的模型和数据集路径仍是 `YOUR_PATH_TO...`，完整 PPL/LongBench/Needle 需要先改成本机真实路径；如果要跑原 CUDA kernel 或 packed quant，需要在 MSVC 编译环境中安装 `kernels` 扩展。

### Llama3.2_3B 对照实验

- 已为 `D:\model\Llama3.2_3B` 补充 Llama3 RoPE 兼容：`experiments/modeling_llama_skvq.py` 现在支持 `rope_scaling.rope_type="llama3"`。
- 新增 `run_llama32_ablation.py`：自动加载本地 Llama3.2_3B，生成/复用本地 minmax reorder cache，并跑 `fp16`、原 SKVQ baseline、方案 A、方案 B、方案 C 的 PPL 对照。
- 已跑 pilot：`seq_len=128, calib=1, eval=1`。由于 `seq_len <= sink + window`，没有触发量化，五组 PPL 相同；该结果仅用于验证流程。
- 已跑有效量化对照：`seq_len=256, window=128, sink=5, calib_samples=2, eval_samples=3`，结果保存到 `experiments/results/llama32_3b/ablation_len256_eval3.csv`。
- 当前小样本结论：量化方法里 `tq_replace` PPL 最低，其次是 `tq_asym_protect`，再是 `tq_hybrid`，原 SKVQ baseline 最高。该结论需要用更大 `eval_samples/seq_len` 复跑确认。
- 已按正式结论设置复跑：`seq_len=512, window=128, sink=5, calib_samples=4, eval_samples=8`，结果保存到 `experiments/results/llama32_3b_formal/ablation_len512_eval8.csv`。
- 正式小规模结论：`tq_asym_protect` 最好，PPL=2.613937，仅比 fp16 高 0.0573%，比原 SKVQ baseline 低 0.8452%；`tq_replace` 次优，PPL=2.630197；`tq_hybrid` 与原 SKVQ 基本持平。
### Extra metrics added on 2026-05-11

- [x] Added `run_llama32_extra_metrics.py` for LongBench-lite, Needle-lite, KV cache compression estimates, and decode latency.
- [x] Wrote LongBench-lite results to `experiments/results/llama32_3b_extra/longbench_lite.csv`.
- [x] Wrote KV cache compression estimates to `experiments/results/llama32_3b_extra/compression_estimates.csv`.
- [x] Wrote fake-quant decode latency to `experiments/results/llama32_3b_extra/latency_decode.csv`.
- [x] Wrote Needle-lite small grid results to `experiments/results/llama32_3b_extra_small/needle_lite.csv`.
- [x] Probed requested 4k/8k Needle points: 4k fp16 depth25 is OK, 8k fp16 depth50 OOM on the 8GB T1000 eager-attention path, so 16k is not practical here without FlashAttention/packed kernel support or more GPU memory.
- [x] Added and ran `run_llama32_wikitext_ablation.py` on WikiText2 test with `seq_len=512`, `eval_samples=16`, `calib_samples=4`; results are in `experiments/results/llama32_3b_wikitext/wikitext2_len512_eval16.csv`.
- [x] WikiText2 PPL changed the ranking from the repeated-text smoke test: `tq_hybrid` is best among quantized methods, then `tq_replace`, then `tq_asym_protect`, all better than the original SKVQ baseline.
- [x] Added and ran `run_llama32_longbench_subset.py` on official `THUDM/LongBench` subset: `passage_retrieval_en`, `lcc`, `gov_report`, 20 examples per task per method, 300 predictions total.
- [x] Official LongBench subset results saved to `experiments/results/llama32_3b_longbench_official_core/summary_n20_ctx1536_gen64.csv` and `details_n20_ctx1536_gen64.csv`.
