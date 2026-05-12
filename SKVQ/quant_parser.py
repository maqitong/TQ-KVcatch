from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import PreTrainedModel

from calib_config import MODEL_TO_REORDER, MODEL_TO_SMOOTH

if TYPE_CHECKING:
    from KVcache_manager import ModelKVCacheManager


def _token_value(tokens: list[str], prefix: str, default=None):
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return default


def _parse_bits(value: str):
    bits = float(value)
    return int(bits) if bits == round(bits) else bits


def get_quantizer_from_str(
    s: str | None,
    model: PreTrainedModel,
    model_name: str,
    *,
    clipping_default: float = 0.92,
    full_prefill_override: bool | None = None,
) -> ModelKVCacheManager | None:
    from KVcache_manager import ModelKVCacheManager

    if s is None or s.strip().lower() == "none":
        return None

    tokens = s.strip().split("-")
    kbits = _parse_bits(tokens[0][1:])
    vbits = _parse_bits(tokens[1][1:])
    gsize = int(_token_value(tokens, "g"))

    if ("rtn" in tokens) or ("rptq" in tokens) or ("smoothquant" in tokens):
        window = 0
    else:
        window = int(_token_value(tokens, "w", 0))

    smooth_file = MODEL_TO_SMOOTH[model_name] if "smooth" in tokens else None
    use_reorder = any(token in tokens for token in ("reorder", "rod", "rptq"))
    reorder_file = (
        MODEL_TO_REORDER[model_name][gsize]["minmax"]
        if use_reorder
        else None
    )

    pre_rope = any(token in tokens for token in ("pre_rope", "rptq", "smoothquant"))
    clip_token = next((token for token in tokens if token.startswith("clip")), None)
    clip_value = clipping_default if clip_token == "clip" else float(clip_token[4:]) if clip_token else 1.0
    clipping = [clip_value for _ in range(len(model.model.layers))]

    if full_prefill_override is None:
        full_prefill = not any(token in tokens for token in ("rtn", "rptq", "smoothquant"))
    else:
        full_prefill = full_prefill_override

    sink_value = _token_value(tokens, "sink", 0)
    protect_value = _token_value(tokens, "protect", 0)
    use_acc_score = float(_token_value(tokens, "h2o", 0))
    use_random = float(_token_value(tokens, "random", 0))

    turboquant_config = None
    if "tq" in tokens:
        turboquant_config = {
            "use_reorder": use_reorder,
            "protected_layers": int(protect_value),
            "seed_base": 42,
        }

    quantizer = ModelKVCacheManager.create(
        model=model,
        kbits=kbits,
        vbits=vbits,
        gsize=gsize,
        window_size=window,
        reorder_file=reorder_file,
        smooth_file=smooth_file,
        clipping=clipping,
        pre_rope=pre_rope,
        full_prefill=full_prefill,
        KIVI_mode="KIVI" in tokens,
        fp8="fp8" in tokens,
        attn_sink=int(sink_value),
        use_acc_score=use_acc_score,
        use_random=use_random,
        turboquant_config=turboquant_config,
    )
    print(f"{'=' * 30}ModelKVManager{'=' * 30}\n{quantizer}")
    return quantizer
