# SKVQ + TurboQuant 集成方案与对比实验设计

> 目标：在 `D:\KVcatch\SKVQ` 工程内实现 **SKVQ + TurboQuant** 的混合 KV Cache 量化方法，并按 SKVQ 论文同样的实验协议跑 PPL / LongBench / Needle-in-Haystack，验证「加上 TurboQuant 后是否有提升」。
>
> 参考实现：
> - SKVQ：`D:\KVcatch\SKVQ`（论文 [arXiv:2405.06219](https://arxiv.org/abs/2405.06219)）
> - TurboQuant：`D:\KVcatch\bt-kvcatch\turboquant-pytorch`（论文 [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)，ICLR 2026）

---

## 1. 两套方法的本质差异（决定如何融合）

| 维度 | SKVQ | TurboQuant (V3) |
|---|---|---|
| 量化器类型 | 仿射均匀量化（asymmetric min-max + zero-point + scale） | 单位球归一化 + 随机正交旋转 + 逐坐标 Lloyd-Max 标量量化 |
| 解决「通道分布不均」的方式 | **离线校准**得到 `reorder_idx` + 按 group 重排 + group-wise quant + `smooth_scale` + per-layer `clipping` | **在线**用 Haar 随机正交矩阵 `Π` 把通道分布旋成各向同性高斯 |
| 长序列处理 | Attention Sink（前 N tok）+ Sliding Window（后 W tok）保留 fp16；中间段量化 | Residual Window（最近 W tok）保留 fp16；之前段量化 |
| K/V 不对称 | KIVI 模式时 K 走 per-channel、V 走 per-token；SKVQ 默认 K/V 同策略 | **显式** `key_bits ≠ value_bits`（如 K4/V2），值域错误经 softmax 被压缩 |
| 层间策略 | 所有层共享同一 `K_bits, V_bits`，只通过 `clipping[layer_idx]` 微调 | **Layer-adaptive**：前 / 后若干层用更高位宽（`protected_layers`） |
| 存储 | bit-packed + fp8 scale/zp | bit-packed indices + fp16 norm |
| 校准 | 必须（reorder + smooth + clipping） | 零校准（只需固定 rotation seed） |

**关键观察**：两者在「滑窗 / sink」上重叠，在「通道处理」上互补——SKVQ 是**离线统计**的解，TurboQuant 是**在线变换**的解。这是融合的核心切入点。

---

## 2. SKVQ 现有改造点定位

把 TurboQuant 接入 SKVQ 只需要改两个文件：

- `D:\KVcatch\SKVQ\KV_process.py` — 量化主体。SKVQuantProcessor.quant_pytorch / quant_cuda 是「量化器」入口，需要在这里增加 TurboQuant 分支。
- `D:\KVcatch\SKVQ\KVcache_manager.py` — sliding window / attention sink / KIVI mode 的调度层，不动；只把 TurboQuant 作为新的"量化后端"插进去。

**整个滑窗 + sink + pre-RoPE + 校准框架完全保留**，只替换/叠加最里层的"对一段 KV 做量化"的 kernel。

---

## 3. 三种递进式集成方案（建议依次实现并对比）

### 方案 A：SKVQ-TQ-Replace（最小改动 / 直接替换）

把 SKVQ 中段（被量化那一段 token）的「仿射 min-max 量化」整体替换为 TurboQuant 的「rotation + Lloyd-Max」。

- 保留：attention sink、sliding window、pre-RoPE、fake_quant / pack 两种 mode
- 移除：reorder、smooth_scale、clipping（这是 SKVQ 用来"逼近"TurboQuant 想做的事的）
- 新增：每层一个 `TurboQuantV3(head_dim, key_bits=k, value_bits=v, residual_window=0, protected_layers=0)`（窗口已经由 SKVQ 的 sliding window 负责，所以 TurboQuant 内部 `residual_window=0`）

**目的**：直接看「TurboQuant 量化器 vs SKVQ 量化器」在同一个 sink+window 调度下谁更强。

### 方案 B：SKVQ-TQ-Hybrid（核心方案 / 互补叠加）

**先 SKVQ reorder，再 TurboQuant rotation**。reorder 把同分布通道聚成 group，然后在每个 group 内做小尺度（`d=gsize`，比如 128 / 64）的 Lloyd-Max。

- 离线校准：复用 SKVQ 现成的 `reorder_idx`（无需重新校准）
- 在线量化：
  1. 滑窗外段 KV 走 reorder（沿用 SKVQ）
  2. group 内 reshape 成 `(N, gsize)`，用 `MSECompressor(head_dim=gsize, bits=...)` 做归一化 + 旋转 + Lloyd-Max
  3. 反量化后逆 reorder 回去
- 旋转矩阵 `Π` 在每层每 group 用不同 seed（成本：每个 group 一个 `gsize×gsize` fp16，~32KB；全模型 < 100MB，可放显存常驻）
- clipping 仍然保留（套在 norm 上，而非 min/max 上：`norm_clipped = clipping × max_norm_in_group`）

**目的**：测试「校准统计（reorder）」和「在线变换（rotation）」是否真的互补。这是论文级别的实验设计——SKVQ 的"reorder 把方差聚类"和 TurboQuant 的"rotation 打散方差"理论上是同一个问题的两种解，叠加未必加分，也可能干扰；这个对比本身就是有价值的发现。

### 方案 C：SKVQ-TQ-Asym（不对称位宽 + 层自适应）

在方案 B 之上引入 TurboQuant 的两个工程技巧：

- **K/V 不对称**：把 SKVQ 现在的 `k2-v2` / `k4-v4` 换成 `k4-v2`、`k6-v2`、`k3-v2`（同等平均位宽下分配更多 bit 给 K）
- **Layer-protected**：前 4 层 + 后 4 层走 `k8-v8`（fp8 直接通过 SKVQ 已有 `fp8` 开关）

**目的**：把 TurboQuant 在 8+ 独立复现里被反复确认的两个工程结论搬过来，看在 SKVQ 框架下是否仍然成立。

---

## 4. 代码改动清单（落地版）

### 4.1 新增文件

```
D:\KVcatch\SKVQ\turboquant_backend.py     # 把 TurboQuant 包成 SKVQ 风格的 quant kernel
```

接口签名（要和 `KV_process.quant_pytorch` 的输入输出对齐）：

```python
class TurboQuantBackend(nn.Module):
    def __init__(self, head_dim, num_layers, key_bits, value_bits,
                 protected_layers=0, protected_bits=8,
                 group_size=None,            # None = per-vector; 设了就是方案 B 的 group 内 rotation
                 use_reorder=False,          # 方案 B 开关
                 seed_base=42):
        ...
    def quant(self, ttype: Literal["k","v"], tensor, layer_idx, reorder_idx=None, group_st_idx=None):
        """
        tensor: [bs, num_heads, seqlen, head_dim]
        return: (fake_dequant_tensor, None, None)  # 暂只支持 fake_quant，对齐 SKVQ 论文实验
        """
        ...
```

实现要点：
- 旋转矩阵 `Π` 用 `turboquant.turboquant.generate_rotation_matrix`，**每层 + 每 group 不同 seed**（`seed = seed_base + layer_idx*1000 + group_idx`），矩阵在 `__init__` 时全部生成并 register_buffer。
- Lloyd-Max 码本用 `turboquant.lloyd_max.LloydMaxCodebook(d=head_dim_or_gsize, bits)`，常驻。
- forward 时 `flat = tensor.reshape(-1, D); norms = flat.norm(...); rotated = (flat / norms) @ Π.T; idx = argmin(|rotated - centroids|); recon = centroids[idx] @ Π * norms`，全程在 fp16 + autocast 下，hot path 不做 .cpu()。
- 关键性能：所有 `Π`、`centroids` 在 init 一次性 stack 成 `(n_layers, [n_groups], D, n_levels)`，forward 用 gather/einsum 一把吃，避免 Python 循环。

### 4.2 改动 `KV_process.py`

在 `SKVQuantProcessor.__init__` 新增参数 `tq_backend: TurboQuantBackend | None = None`。在 `quantization` 里加一个分支：

```python
if self.tq_backend is not None:
    fake, _, _ = self.tq_backend.quant(ttype, tensor, self.layer_idx,
                                       self.reorder_idx, self.group_st_idx)
    return fake, None, None
```

放在 `if impl == "py"` 之前，确保 TurboQuant 优先。

### 4.3 改动 `KVcache_manager.py`

`ModelKVCacheManager.create(...)` 增加可选关键字 `turboquant_config: dict | None = None`，构造时把 `TurboQuantBackend(**turboquant_config)` 注入到 `processor_config["tq_backend"]`。

### 4.4 改动评测脚本

把 `--quant` 字符串扩展出新关键字：

| 量化字符串 | 含义 |
|---|---|
| `k2-v2-w128-g128-reorder-pre_rope-clip-sink5` | 原版 SKVQ（基线） |
| `k2-v2-w128-g128-KIVI` | 原版 KIVI（基线） |
| `k4-v4-tq-w128-sink5` | 方案 A：纯 TurboQuant 替换 |
| `k4-v4-tq-w128-g128-reorder-sink5` | 方案 B：reorder + TurboQuant group rotation |
| `k4-v2-tq-w128-g128-reorder-sink5-protect4` | 方案 C：不对称 + 层保护 |

解析逻辑放在 `eval_ppl.py` / `eval_longbench.py` / `eval_needle.py` 三个脚本里复用一个 `parse_quant_str()`。

### 4.5 当前落地实现记录

已按最小改动原则先实现 **fake-quant 精度实验版**：

- 新增 `D:\KVcatch\SKVQ\turboquant_backend.py`：提供 `TurboQuantBackend`，输出仍是 `[bs, heads, seqlen, head_dim]` 的 fake-dequant tensor，不改 SKVQ cache tuple 格式。
- 新增 `D:\KVcatch\SKVQ\quant_parser.py`：统一解析 `--quant` 字符串，`eval_longbench.py` / `eval_needle.py` 已切到该 helper，`eval_ppl.py` 新增 `--quant` 单配置入口。
- `KV_process.py`：`SKVQuantProcessor` 增加可选 `tq_backend`，当该字段存在时优先走 TurboQuant fake-quant。
- `KVcache_manager.py`：`ModelKVCacheManager.create(..., turboquant_config=...)` 会为每层创建一个 TurboQuant 后端，并在 `tag()` 中追加 `-tq` / `-tqrod`。
- 当前 TurboQuant 后端支持整数 bit（如 `k4-v4`、`k4-v2`），暂不支持 `1.5bit`；`k1.5-v1.5` 仍走原 SKVQ。
- 当前后端优先复用 `D:\KVcatch\bt-kvcatch\turboquant-pytorch` 的 rotation / Lloyd-Max；如果本机环境缺少 `scipy`，会退回到内置的高斯 Lloyd-Max fallback，避免额外依赖阻塞小测试。

推荐实际运行字符串改为和现有 SKVQ parser 兼容的顺序：

| 量化字符串 | 含义 |
|---|---|
| `k2-v2-g128-w128-reorder-pre_rope-clip-sink5-fp8` | 原版 SKVQ |
| `k4-v4-g128-w128-tq-sink5` | 方案 A：TurboQuant 替换 |
| `k4-v4-g128-w128-tq-reorder-pre_rope-clip-sink5` | 方案 B：reorder + TurboQuant |
| `k4-v2-g128-w128-tq-reorder-pre_rope-clip-sink5-protect4` | 方案 C：K/V 不对称 + 层保护 |

---

## 5. 对比实验设计（对齐 SKVQ 论文）

### 5.1 模型矩阵

复用 SKVQ 论文用的模型（在 `calib_config.py` 已注册），优先做这三档：

| 模型 | 用途 | 已有校准产物 |
|---|---|---|
| LLaMA2-7B | PPL on WikiText2 / PTB（短文本基础对比） | 有 |
| LLaMA2-7B-80k | Needle-in-Haystack @ 32K ctx（长上下文） | 有 |
| Mistral-7B-Instruct-v0.2 | LongBench（综合下游任务） | 有 |

（资源足够再加 LLaMA3-70B-Instruct 复现 LongBench；不够就跳过。）

### 5.2 量化位宽配置矩阵

| ID | Config | 平均位宽 | 备注 |
|---|---|---|---|
| C1 | `k4-v4` | 4.0 | 简单档 |
| C2 | `k3-v3` | 3.0 | SKVQ 论文主结果 |
| C3 | `k2-v2` | 2.0 | SKVQ 论文极致档 |
| C4 | `k4-v2` | 3.0 | 非对称版本（同 C2 平均位宽） |
| C5 | `k1.5-v1.5` | 1.5 | SKVQ 极限挑战（论文有） |

每个 Config 跑 4 个方法：

1. **B0**：fp16（baseline）
2. **B1**：SKVQ 原版（论文设置：`reorder + pre_rope + clip + sink5 + window=128 + group=128`）
3. **M1**：方案 A（SKVQ-TQ-Replace）
4. **M2**：方案 B（SKVQ-TQ-Hybrid，**主对比对象**）
5. **M3**：方案 C（仅在 C4 / 异步组合下跑，验证 K/V 不对称增益）

### 5.3 评测指标（直接复用 SKVQ 现成脚本）

| Benchmark | 脚本 | 指标 | 主要看 |
|---|---|---|---|
| PPL on WikiText2 | `eval_ppl.py` | perplexity（越低越好） | C1/C2/C3 下 M2 vs B1 的差 |
| LongBench (16 子任务) | `eval_longbench.py` + `score_longbench.py` | 每任务分数 + 平均 | 平均分 + Single-Doc QA + Multi-Doc QA + Summarization |
| Needle-in-Haystack | `eval_needle.py` + `viz-needle.ipynb` | 命中率热力图 | 32K ctx 下命中率 |

**核心结论判定**（在写论文 / 总结时用这套）：

> M2（SKVQ-TQ-Hybrid）相对 B1（原 SKVQ）：
> - **PPL 下降 ≥ 1%** 且 **LongBench 平均 ↑ ≥ 0.5 分** 且 **Needle 命中率不降** → 有提升
> - 任一条不满足 → 列出退化任务，分析是 rotation 干扰了 reorder 还是位宽不够

### 5.4 控制变量清单（一致性要求）

跑每组对比时严格对齐：

- 同一份 model checkpoint（`MODEL_NAME_TO_PATH`）
- 同一份 reorder 文件（`MODEL_TO_REORDER[model][gsize]["minmax"]`）— 方案 A 也用同样的 group 划分但不做 reorder
- 同一份 clipping（默认 0.95，可在 0.91–0.97 之间扫一遍）
- `attention_sink=5, window_size=128, group_size=128, pre_rope=True`
- `fake_quant=True`（与 SKVQ 论文 reported number 对齐；pack 模式只在最后做端到端 latency 测试）
- 随机种子：TurboQuant 的 `Π` 和 Lloyd-Max 在 `seed=42` 下固定，避免单次随机的方差污染

### 5.5 消融实验（如果时间允许，挑 2–3 个跑）

| Ablation | 改什么 | 想回答什么 |
|---|---|---|
| A1 | M2 关掉 reorder（变成纯 group 内 rotation） | reorder 在加了 rotation 后还有用吗 |
| A2 | M2 关掉 clipping | rotation 后 outlier 是否被自然处理掉了 |
| A3 | C4 (k4-v2) vs C2 (k3-v3) 同平均位宽对比 | 不对称分配在 SKVQ 框架下是否成立 |
| A4 | Π 共享 vs 每层独立 | 旋转多样性的价值 |

---

## 6. 实施时间线建议（单人 / 单卡 7–10 天）

| Day | 任务 |
|---|---|
| 1 | 在 SKVQ 工程里跑通原版 B1（C2 配置）确认环境 + 拉通 LongBench 一小组子任务 |
| 2 | 实现 `turboquant_backend.py`（方案 A），写 toy test 验证 dequant 精度 vs reference TurboQuant |
| 3 | 接入 `KV_process.py`，跑 PPL 方案 A，看是否 work |
| 4 | 实现方案 B（group 内 rotation），跑 PPL + 1 个 LongBench 子任务 |
| 5 | 跑全 LongBench（M2 vs B1）on Mistral-7B |
| 6 | 跑 Needle on LLaMA2-7B-80k @ 32K |
| 7 | 实现方案 C，跑 C4 / A3 消融 |
| 8 | 跑 A1 / A2 消融 + clipping 扫一遍 |
| 9 | 整理结果表格，画图，写实验报告 |
| 10 | buffer / 复跑 outlier 数据点 |

---

## 7. 风险与备选

| 风险 | 缓解 |
|---|---|
| reorder + rotation 互相干扰，M2 反而比 B1 差 | 这本身是 valid finding，写进 ablation 报告即可；并退到方案 A 的纯替换实验 |
| group 内 rotation 的小尺度（d=128）使 Lloyd-Max 近似不够好 | 1) 切换到更大 d（取消 group，做 head_dim 全量旋转，对应方案 A）；2) 用 TurboQuant 论文里的 `optimal_distortion` 函数验证理论下界 |
| TurboQuant 的 fp32 旋转开销在 prefill 阶段慢 | 仅做 fake_quant 实验（论文设置允许）；推理速度对比放最后单独做，不阻塞精度对比 |
| 长上下文（80k）显存炸 | 先跑 4k → 16k → 32k；fake_quant 模式下显存与原 SKVQ 完全一致，不会炸新 |
| TurboQuant 的 V3 在 `residual_window=0` 下被官方 README 标注"不稳定" | 由 SKVQ 的 sliding window 提供等价 residual_window；TurboQuant 内部一律 `residual_window=0` |

---

## 8. 输出物清单

跑完后需要交付：

1. `D:\KVcatch\SKVQ\turboquant_backend.py`（新增）
2. `D:\KVcatch\SKVQ\KV_process.py` / `KVcache_manager.py`（小改）
3. `D:\KVcatch\SKVQ\experiments\results\` 目录下：
   - `ppl_results.csv`（model × config × method → ppl）
   - `longbench_results.csv`（model × config × method × task → score）
   - `needle_<model>_<config>_<method>.png` 命中率热力图
4. `D:\KVcatch\SKVQ_plus_TurboQuant_实验报告.md`（最终汇报）

---

## 9. 立即可以开始的第一步

1. 在 `D:\KVcatch\SKVQ` 装环境（`pip install -r requirements.txt && cd kernels && pip install -e .`），跑通 `python eval_ppl.py --model llama2-7b` 拿到 B1 的 PPL 数字。
2. 把 `D:\KVcatch\bt-kvcatch\turboquant-pytorch\turboquant` 整个目录 `pip install -e .` 装成 package（或者直接在 SKVQ 里 `sys.path` 加进去），import 测试 `from turboquant.compressors_v3 import MSECompressor` 成功。
3. 开始写 `turboquant_backend.py`，先实现方案 A 的最小可运行版本。

只要第 1、2 步通了，剩下的工作就是按上面表格往里填数。
