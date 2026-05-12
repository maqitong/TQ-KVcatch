# Key Norm、Attention Score、Softmax 的关系与区别

## Key Norm 是什么

Key norm 就是 Key 向量的 **L2 范数**，即向量的"长度"：

$$\|k\|_2 = \sqrt{k_1^2 + k_2^2 + \cdots + k_D^2}$$

它是一个**标量**（一个数），表示这个 token 的 Key 向量有多"大"。

在 TurboQuant 项目代码里对应：

```python
vec_norms = torch.norm(flat, dim=-1)  # 每个 token 的 K 向量算一个长度标量
```

---

## Key Norm 的作用

在 TurboQuant 量化流程里，key norm 起的是**尺度分离**的作用：

```
原始 K 向量  =  方向（单位球上的点） × 长度（norm）
                      ↑ 用码本量化           ↑ 用 fp16 单独存
```

- **压缩时**：把向量除以 norm 归一化到单位球，再旋转 + 量化方向，norm 单独存一个 fp16
- **解压时**：把方向从码本还原，再乘回 norm，恢复原始向量

如果不存 norm：解压出的向量长度全是 1，原始幅度信息丢失，attention score 全部偏掉。

---

## 三者的关系：所处的不同阶段

这三个概念在 attention 计算里是**三个完全不同阶段**：

```
K 向量（含 norm）
        ↓
① 内积打分：score = Q · K^T / √D     ← attention score
        ↓
② 归一化：weight = softmax(score)     ← softmax
        ↓
③ 加权求和：output = weight · V
```

---

## 三者区别

| 概念 | 是什么 | 发生在哪一步 | 形状 |
|---|---|---|---|
| **Key norm** | K 向量的 L2 长度 | 量化/解压时的尺度还原 | 每 token 一个标量 |
| **Attention score** | Q · K^T 的内积结果 | 所有 K 计算完后 | `(seq_q, seq_k)` 矩阵 |
| **Softmax** | 对 score 做指数归一化 | score 算完之后 | 同上，值在 0~1 之间 |

Key norm 影响 attention score 的方式：

$$\langle q, k \rangle = \|q\| \cdot \|k\| \cdot \cos\theta$$

norm 越大，内积绝对值越大，对应位置的 score 越大。

---

## 为什么 K/V norm 不对称很重要

Qwen 模型实测：Key norm 在 **172~778**，Value norm 只有 **2~4**，相差约 100 倍。

这意味着：

- Key 向量很"长"，量化误差被 norm 放大后对 score 影响巨大
- score 偏差大 → softmax 指数放大 → attention 权重全乱 → 生成乱码
- 因此 V3 采用非对称 bit：**K 用更多 bit 保精度，V 可以少 bit**
