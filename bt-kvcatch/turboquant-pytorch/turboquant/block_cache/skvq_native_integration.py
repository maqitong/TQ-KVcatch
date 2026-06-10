"""SKVQ paper ``skvq_baseline`` path (ModelKVCacheManager + LlamaForCausalLM)."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

# Paper-aligned defaults (SKVQ run_exp1 / run_exp2).
PAPER_SINK = 5
PAPER_WINDOW = 128
PAPER_CLIP = 0.96
PAPER_GROUP_SIZE = 128

# Align ``block_tq_pure_mix`` with main ``Hybrid+TQ+Block+PageMix`` (run_gpu0.sh).
PAPER_MIX_PROTECTED_LAYERS = 1
PAPER_MIX_PROTECTED_KEY_BITS = 8
PAPER_MIX_PROTECTED_VALUE_BITS = 8


def paper_pure_layer_protection(
    paper_baseline: str | None,
    args: Any,
) -> tuple[int, float, float]:
    """Resolve paper-style protected-layer settings for pure block baselines."""
    if paper_baseline == "tq_pure_mix":
        layers_arg = int(getattr(args, "protected_layers", 0))
        if layers_arg < 0:
            return 0, float(PAPER_MIX_PROTECTED_KEY_BITS), float(
                PAPER_MIX_PROTECTED_VALUE_BITS
            )
        if layers_arg == 0:
            return (
                PAPER_MIX_PROTECTED_LAYERS,
                float(PAPER_MIX_PROTECTED_KEY_BITS),
                float(PAPER_MIX_PROTECTED_VALUE_BITS),
            )
        layers = layers_arg
        key_bits = getattr(args, "protected_key_bits", PAPER_MIX_PROTECTED_KEY_BITS)
        value_bits = getattr(args, "protected_value_bits", PAPER_MIX_PROTECTED_VALUE_BITS)
        return layers, float(key_bits), float(value_bits)
    return 0, float(getattr(args, "protected_key_bits", 8)), float(
        getattr(args, "protected_value_bits", 8)
    )


def resolve_skvq_root(explicit: str | None = None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"SKVQ root not found: {root}")
        return root
    env = os.environ.get("SKVQ_ROOT", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if root.is_dir():
            return root
    here = Path(__file__).resolve()
    for base in (here.parents[4], here.parents[3], Path("/root/autodl-tmp/bt-kvcatch")):
        candidate = base / "SKVQ"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "SKVQ repo not found; set SKVQ_ROOT or place SKVQ next to turboquant-pytorch"
    )


def _ensure_skvq_importable(skvq_root: Path) -> None:
    root = str(skvq_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def enable_skvq_generation(skvq_root: Path | None = None) -> None:
    """Patch SKVQ LlamaForCausalLM with transformers GenerationMixin (4.5x+)."""
    root = resolve_skvq_root(skvq_root)
    _ensure_skvq_importable(root)
    import experiments.modeling_llama_skvq as modeling_llama_skvq
    from transformers.generation.utils import GenerationMixin

    if hasattr(modeling_llama_skvq.LlamaForCausalLM, "generate"):
        return

    class LlamaForCausalLM(modeling_llama_skvq.LlamaForCausalLM, GenerationMixin):
        pass

    modeling_llama_skvq.LlamaForCausalLM = LlamaForCausalLM


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


@torch.no_grad()
def skvq_native_generate(
    model,
    tokenizer,
    inputs: dict,
    *,
    max_new_tokens: int,
) -> torch.LongTensor:
    """Run generate and return full token ids [batch, seq]."""
    device = _model_device(model)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    if hasattr(output, "sequences"):
        return output.sequences
    return output


def load_skvq_llama(
    model_path: str,
    *,
    skvq_root: str | None = None,
    dtype: torch.dtype | None = None,
    device_map: str = "auto",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    use_flash_attn: bool = True,
):
    root = resolve_skvq_root(skvq_root)
    enable_skvq_generation(root)
    from experiments.modeling_llama_skvq import LlamaForCausalLM
    from transformers import LlamaConfig
    from transformers.utils import is_flash_attn_2_available

    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    flash_on = use_flash_attn and is_flash_attn_2_available()
    if flash_on:
        config._flash_attn_2_enabled = True

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def detach_quantizer(model) -> None:
    for layer in model.model.layers:
        layer.self_attn.KV_cache_manager = None
    model.model.model_kv_manager = None
    model.model_kv_manager = None


def clear_quantizer(model) -> None:
    manager = getattr(model.model, "model_kv_manager", None)
    if manager is not None:
        manager.clear()


def build_skvq_baseline_manager(
    model,
    *,
    reorder_file: str | Path,
    key_bits: float = 2,
    value_bits: float = 2,
    group_size: int = PAPER_GROUP_SIZE,
    window_size: int = PAPER_WINDOW,
    sink: int = PAPER_SINK,
    clip: float = PAPER_CLIP,
    skvq_root: str | None = None,
):
    root = resolve_skvq_root(skvq_root)
    _ensure_skvq_importable(root)
    from KVcache_manager import ModelKVCacheManager

    num_layers = len(model.model.layers)
    clipping = [clip] * num_layers
    return ModelKVCacheManager.create(
        model=model,
        kbits=key_bits,
        vbits=value_bits,
        gsize=group_size,
        reorder_file=str(reorder_file),
        smooth_file=None,
        window_size=window_size,
        pre_rope=True,
        clipping=clipping,
        attn_sink=sink,
        full_prefill=False,
        fp8=True,
        fake_quant=True,
    )


def plug_quantizer(model, manager) -> None:
    root = resolve_skvq_root()
    _ensure_skvq_importable(root)
    from experiments.utils import plug_quantizer_into_model

    plug_quantizer_into_model(model, manager)


def skvq_native_config(
    *,
    key_bits: float = 2,
    value_bits: float = 2,
    reorder_file: str | None = None,
) -> dict:
    return {
        "policy": "sliding_window",
        "window": PAPER_WINDOW,
        "sink": PAPER_SINK,
        "key_bits": key_bits,
        "value_bits": value_bits,
        "group_size": PAPER_GROUP_SIZE,
        "clip": PAPER_CLIP,
        "protected_layers": 0,
        "reorder": True,
        "integration": "SKVQ/ModelKVCacheManager",
        "reorder_file": reorder_file,
    }
