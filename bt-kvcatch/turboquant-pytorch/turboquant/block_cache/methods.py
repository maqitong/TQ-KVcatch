"""Shared backend/method construction for evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import BlockCacheConfig
from .hf_cache import BlockKVCache
from .policies import HybridPolicy, TokenBlockPolicy, WindowBlockPolicy


NIAH_ALL_BACKENDS = [
    "dynamic",
    "block_tq",
    "block_tq_mix",
    "block_skvq",
    "block_skvq_mix",
]

PPL_ALL_BACKENDS = [
    *NIAH_ALL_BACKENDS,
    "block_tq_pure",
    "block_tq_pure_mix",
    "v3_flat",
]

LONGBENCH_ALL_BACKENDS = [
    *PPL_ALL_BACKENDS,
    "skvq_native",
]

BLOCK_BACKENDS = {
    "block_tq",
    "block_tq_mix",
    "block_skvq",
    "block_skvq_mix",
    "block_tq_pure",
    "block_tq_pure_mix",
}


@dataclass
class MethodSpec:
    """Formal experiment method definition.

    ``backend`` names the runtime path, while ``quant_backend`` names the
    page compressor used by ``BlockKVCache`` when applicable.
    """

    name: str
    backend: str
    quant_backend: str | None
    policy: str
    method_group: str = "method"
    page_quant_scheme: str = "none"
    mixed_precision: bool = False
    importance_metric: str = "k_norm"
    key_bits: float = 2
    value_bits: float = 2
    high_key_bits: float = 4
    high_value_bits: float = 4
    low_key_bits: float = 2
    low_value_bits: float = 2
    paper_baseline: str | None = None


DEFAULT_MIX_HIGH_KEY_BITS = 4.0
DEFAULT_MIX_HIGH_VALUE_BITS = 4.0
DEFAULT_PURE_MIX_HIGH_KEY_BITS = 3.0
DEFAULT_PURE_MIX_HIGH_VALUE_BITS = 3.0
DEFAULT_MIX_LOW_KEY_BITS = 2.0
DEFAULT_MIX_LOW_VALUE_BITS = 2.0


def _arg_or_default(args: Any, name: str, default: float) -> float:
    value = getattr(args, name, None)
    return float(default if value is None else value)


def _mix_bits(args: Any, *, pure_mix: bool = False) -> tuple[float, float, float, float]:
    high_k_default = (
        DEFAULT_PURE_MIX_HIGH_KEY_BITS if pure_mix else DEFAULT_MIX_HIGH_KEY_BITS
    )
    high_v_default = (
        DEFAULT_PURE_MIX_HIGH_VALUE_BITS if pure_mix else DEFAULT_MIX_HIGH_VALUE_BITS
    )
    return (
        _arg_or_default(args, "high_key_bits", high_k_default),
        _arg_or_default(args, "high_value_bits", high_v_default),
        _arg_or_default(args, "low_key_bits", DEFAULT_MIX_LOW_KEY_BITS),
        _arg_or_default(args, "low_value_bits", DEFAULT_MIX_LOW_VALUE_BITS),
    )


def _mix_scheme(high_k: float, high_v: float, low_k: float, low_v: float) -> str:
    return f"Mixed high K{high_k}/V{high_v}, low K{low_k}/V{low_v}"


def parse_backend_selection(name: str, *, all_backends: list[str]) -> list[str]:
    if name == "all":
        return list(all_backends)
    if "," in name:
        return [backend.strip() for backend in name.split(",") if backend.strip()]
    return [name]


def build_policy(args: Any, *, window_uses_sink: bool = True):
    if args.policy == "token":
        return TokenBlockPolicy()
    if args.policy == "window":
        sink_size = args.sink if window_uses_sink else 0
        return WindowBlockPolicy(window_size=args.window, sink_size=sink_size)
    if args.policy == "hybrid":
        return HybridPolicy(sink_size=args.sink, window_size=args.window)
    raise ValueError(f"unknown policy: {args.policy}")


def policy_from_name(name: str, args: Any):
    if name == "token":
        return TokenBlockPolicy()
    if name == "hybrid":
        return HybridPolicy(sink_size=args.sink, window_size=args.window)
    raise ValueError(f"unknown method policy: {name}")


def paper_tq_pure_policy():
    from turboquant.block_cache.skvq_native_integration import PAPER_SINK, PAPER_WINDOW

    return HybridPolicy(sink_size=PAPER_SINK, window_size=PAPER_WINDOW)


def build_main_methods(args: Any) -> list[MethodSpec]:
    """Build the formal comparison method list used by ``experiment_main``."""
    baseline_scheme = f"Uniform K{args.key_bits}/V{args.value_bits}"
    high_k, high_v, low_k, low_v = _mix_bits(args)
    pure_high_k, pure_high_v, pure_low_k, pure_low_v = _mix_bits(args, pure_mix=True)
    mix_scheme = _mix_scheme(high_k, high_v, low_k, low_v)
    pure_mix_scheme = _mix_scheme(pure_high_k, pure_high_v, pure_low_k, pure_low_v)
    methods = [
        MethodSpec(
            name="FP16",
            backend="dynamic",
            quant_backend=None,
            policy="none",
            method_group="reference",
            page_quant_scheme="FP16",
        ),
        MethodSpec(
            name="SKVQ Baseline",
            backend="block_skvq",
            quant_backend="skvq",
            policy="token",
            method_group="baseline",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
        ),
        MethodSpec(
            name="TurboQuant Baseline",
            backend="block_tq",
            quant_backend="turboquant",
            policy="token",
            method_group="baseline",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
        ),
        MethodSpec(
            name="SKVQ skvq_baseline (native)",
            backend="skvq_native",
            quant_backend="skvq",
            policy="window",
            method_group="paper_baseline",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            paper_baseline="skvq_native",
        ),
        MethodSpec(
            name="TurboQuant V3 flat (rw=128, K2/V2)",
            backend="v3_flat",
            quant_backend=None,
            policy="none",
            method_group="paper_baseline",
            page_quant_scheme=f"V3 flat K{args.key_bits}/V{args.value_bits}",
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            paper_baseline="v3_flat",
        ),
        MethodSpec(
            name="TurboQuant pure (tq_replace)",
            backend="block_tq",
            quant_backend="turboquant",
            policy="hybrid",
            method_group="paper_baseline",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            paper_baseline="tq_pure",
        ),
        MethodSpec(
            name="TurboQuant pure+PageMix",
            backend="block_tq_pure_mix",
            quant_backend="turboquant",
            policy="hybrid",
            method_group="paper_baseline",
            page_quant_scheme=pure_mix_scheme,
            mixed_precision=True,
            importance_metric=args.importance_metric,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            high_key_bits=pure_high_k,
            high_value_bits=pure_high_v,
            low_key_bits=pure_low_k,
            low_value_bits=pure_low_v,
            paper_baseline="tq_pure_mix",
        ),
        MethodSpec(
            name="Hybrid+SKVQ+Block",
            backend="block_skvq",
            quant_backend="skvq",
            policy="hybrid",
            method_group="method",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
        ),
        MethodSpec(
            name="Hybrid+TQ+Block",
            backend="block_tq",
            quant_backend="turboquant",
            policy="hybrid",
            method_group="method",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
        ),
        MethodSpec(
            name="Hybrid+TQ+Block+PageMix",
            backend="block_tq_mix",
            quant_backend="turboquant",
            policy="hybrid",
            method_group="method",
            page_quant_scheme=mix_scheme,
            mixed_precision=True,
            importance_metric=args.importance_metric,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            high_key_bits=high_k,
            high_value_bits=high_v,
            low_key_bits=low_k,
            low_value_bits=low_v,
        ),
    ]
    if args.include_random_mix:
        methods.append(
            MethodSpec(
                name="Hybrid+TQ+RandomMix",
                backend="block_tq_random_mix",
                quant_backend="turboquant",
                policy="hybrid",
                method_group="ablation",
                page_quant_scheme=mix_scheme,
                mixed_precision=True,
                importance_metric="random",
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                high_key_bits=high_k,
                high_value_bits=high_v,
                low_key_bits=low_k,
                low_value_bits=low_v,
            )
        )
    return methods


def _block_backend_config(
    args: Any,
    backend: str,
    *,
    window_uses_sink: bool = True,
) -> tuple[BlockCacheConfig, str | None]:
    quant_backend = "skvq" if "skvq" in backend else "turboquant"
    mixed = backend.endswith("_mix")

    if backend in ("block_tq_pure", "block_tq_pure_mix"):
        from turboquant.block_cache.skvq_native_integration import (
            PAPER_CLIP,
            paper_pure_layer_protection,
        )

        policy = paper_tq_pure_policy()
        paper_tag = "tq_pure_mix" if backend == "block_tq_pure_mix" else "tq_pure"
        protected_layers, prot_k, prot_v = paper_pure_layer_protection(paper_tag, args)
        reorder_file = None
        clipping = PAPER_CLIP
    else:
        policy = build_policy(args, window_uses_sink=window_uses_sink)
        reorder_file = args.reorder_file
        protected_layers = args.protected_layers
        prot_k = args.protected_key_bits
        prot_v = args.protected_value_bits
        clipping = args.clipping
    high_k, high_v, low_k, low_v = _mix_bits(args, pure_mix=backend == "block_tq_pure_mix")

    cfg = BlockCacheConfig(
        block_size=args.block_size,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        granularity=args.granularity,
        policy=policy,
        quant_backend=quant_backend,
        mixed_precision=mixed,
        importance_metric=args.importance_metric,
        important_ratio=args.important_ratio,
        high_key_bits=high_k,
        high_value_bits=high_v,
        low_key_bits=low_k,
        low_value_bits=low_v,
        num_layers=args.num_layers,
        protected_layers=protected_layers,
        protected_key_bits=prot_k,
        protected_value_bits=prot_v,
        group_size=args.group_size,
        key_group_size=args.key_group_size,
        value_group_size=args.value_group_size,
        clipping=clipping,
        reorder_file=reorder_file,
        max_cached_decompressed_blocks=args.max_cached_decompressed_blocks,
        incremental_materialize=getattr(args, "incremental_materialize", False),
        quant_budget_per_update=getattr(args, "quant_budget_per_update", None),
    )
    paper_baseline = (
        "tq_pure_mix"
        if backend == "block_tq_pure_mix"
        else ("tq_pure" if backend == "block_tq_pure" else None)
    )
    return cfg, paper_baseline


def cache_factory_for_backend(
    args: Any,
    backend: str,
    *,
    include_v2: bool = False,
    include_v3: bool = False,
    include_skvq_native: bool = False,
    window_uses_sink: bool = True,
    v3_protected_layers: int | None = None,
) -> Callable[[], Any]:
    if backend == "dynamic":
        return lambda: None
    if backend == "skvq_native" and include_skvq_native:
        return lambda: None

    if backend == "v2_paper" and include_v2:
        from turboquant.block_cache.v2_paper_cache import V2PaperCache

        def make_v2() -> V2PaperCache:
            return V2PaperCache(
                key_bits=int(args.key_bits),
                value_bits=int(args.value_bits),
                n_layers=int(args.num_layers or 32),
                seed=42,
            )

        return make_v2

    if backend == "v3_flat" and include_v3:
        from turboquant.block_cache.v3_flat_cache import V3FlatCache

        def make_v3() -> V3FlatCache:
            protected_layers = (
                int(getattr(args, "protected_layers", 0))
                if v3_protected_layers is None
                else int(v3_protected_layers)
            )
            return V3FlatCache(
                key_bits=int(args.key_bits),
                value_bits=int(args.value_bits),
                residual_window=int(getattr(args, "residual_window", 128)),
                protected_layers=protected_layers,
                n_layers=int(args.num_layers or 32),
            )

        return make_v3

    if backend not in BLOCK_BACKENDS:
        raise ValueError(f"unknown backend: {backend}")

    def make_cache() -> BlockKVCache:
        cfg, _paper_baseline = _block_backend_config(
            args,
            backend,
            window_uses_sink=window_uses_sink,
        )
        return BlockKVCache(cfg)

    return make_cache


def cache_factory_for_method(args: Any, method: MethodSpec) -> Callable[[], Any]:
    """Build the cache factory for a formal ``MethodSpec``."""
    if method.backend == "skvq_native":
        return lambda: None

    if method.backend == "v3_flat":
        from turboquant.block_cache.v3_flat_cache import V3FlatCache

        def make_v3() -> V3FlatCache:
            return V3FlatCache(
                key_bits=int(method.key_bits),
                value_bits=int(method.value_bits),
                residual_window=int(args.residual_window),
                protected_layers=0,
                n_layers=int(args.num_layers or 32),
                seed=42,
            )

        return make_v3

    if method.quant_backend is None:
        return lambda: None

    def make_cache() -> BlockKVCache:
        if method.paper_baseline in ("tq_pure", "tq_pure_mix"):
            from turboquant.block_cache.skvq_native_integration import (
                PAPER_CLIP,
                paper_pure_layer_protection,
            )

            policy = paper_tq_pure_policy()
            reorder_file = None
            protected_layers, prot_k, prot_v = paper_pure_layer_protection(
                method.paper_baseline, args
            )
            clipping = PAPER_CLIP
        else:
            prot_k = args.protected_key_bits
            prot_v = args.protected_value_bits
            policy = policy_from_name(method.policy, args)
            reorder_file = args.reorder_file
            protected_layers = args.protected_layers
            clipping = args.clipping

        cfg = BlockCacheConfig(
            block_size=args.block_size,
            key_bits=method.key_bits,
            value_bits=method.value_bits,
            granularity=args.granularity,
            policy=policy,
            quant_backend=method.quant_backend,
            mixed_precision=method.mixed_precision,
            importance_metric=method.importance_metric,
            important_ratio=args.important_ratio,
            high_key_bits=method.high_key_bits,
            high_value_bits=method.high_value_bits,
            low_key_bits=method.low_key_bits,
            low_value_bits=method.low_value_bits,
            num_layers=args.num_layers,
            protected_layers=protected_layers,
            protected_key_bits=prot_k,
            protected_value_bits=prot_v,
            group_size=args.group_size,
            key_group_size=args.key_group_size,
            value_group_size=args.value_group_size,
            clipping=clipping,
            reorder_file=reorder_file,
            max_cached_decompressed_blocks=args.max_cached_decompressed_blocks,
            incremental_materialize=getattr(args, "incremental_materialize", False),
            quant_budget_per_update=getattr(args, "quant_budget_per_update", None),
        )
        return BlockKVCache(cfg)

    return make_cache


def backend_config(args: Any, backend: str) -> dict[str, Any]:
    """Return the human-readable method config used in result tables."""
    if backend == "v2_paper":
        return {
            "policy": "v2_paper",
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "mixed": False,
            "integration": "turboquant/V2PaperCache+CompressorV2(QJL)",
        }
    if backend == "v3_flat":
        return {
            "policy": "v3_flat",
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "residual_window": int(getattr(args, "residual_window", 128)),
            "mixed": False,
            "paper_baseline": "v3_flat",
            "integration": "turboquant/V3FlatCache+MSECompressor",
        }
    if backend in ("block_tq_pure", "block_tq_pure_mix"):
        from turboquant.block_cache.skvq_native_integration import (
            PAPER_CLIP,
            PAPER_SINK,
            PAPER_WINDOW,
            paper_pure_layer_protection,
        )

        mixed = backend == "block_tq_pure_mix"
        paper_tag = "tq_pure_mix" if mixed else "tq_pure"
        prot_layers, prot_k, prot_v = paper_pure_layer_protection(paper_tag, args)
        high_k, high_v, low_k, low_v = _mix_bits(args, pure_mix=mixed)
        return {
            "policy": "hybrid",
            "block_size": args.block_size,
            "sink": PAPER_SINK,
            "window": PAPER_WINDOW,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "mixed": mixed,
            "important_ratio": args.important_ratio if mixed else None,
            "importance_metric": args.importance_metric if mixed else None,
            "high_key_bits": high_k if mixed else None,
            "high_value_bits": high_v if mixed else None,
            "low_key_bits": low_k if mixed else None,
            "low_value_bits": low_v if mixed else None,
            "paper_baseline": paper_tag,
            "reorder": False,
            "protected_layers": prot_layers,
            "protected_key_bits": prot_k if mixed else None,
            "protected_value_bits": prot_v if mixed else None,
            "clipping": PAPER_CLIP,
            "quant_budget_per_update": getattr(args, "quant_budget_per_update", None),
            "integration": "turboquant-pytorch/BlockKVCache",
        }
    if backend == "skvq_native":
        from turboquant.block_cache.skvq_native_integration import skvq_native_config

        return skvq_native_config(
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            reorder_file=args.reorder_file,
        )
    high_k, high_v, low_k, low_v = _mix_bits(args)
    return {
        "policy": args.policy,
        "block_size": args.block_size,
        "sink": args.sink,
        "window": args.window,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "mixed": backend.endswith("_mix"),
        "importance_metric": args.importance_metric,
        "important_ratio": args.important_ratio,
        "high_key_bits": high_k,
        "high_value_bits": high_v,
        "low_key_bits": low_k,
        "low_value_bits": low_v,
        "num_layers": args.num_layers,
        "protected_layers": args.protected_layers,
        "protected_key_bits": args.protected_key_bits,
        "protected_value_bits": args.protected_value_bits,
        "group_size": args.group_size,
        "key_group_size": args.key_group_size,
        "value_group_size": args.value_group_size,
        "max_cached_decompressed_blocks": args.max_cached_decompressed_blocks,
        "incremental_materialize": getattr(args, "incremental_materialize", False),
        "quant_budget_per_update": getattr(args, "quant_budget_per_update", None),
        "integration": "turboquant-pytorch/BlockKVCache",
    }


def method_config(args: Any, method: MethodSpec) -> dict[str, Any]:
    """Return the human-readable config for a formal ``MethodSpec`` result."""
    if method.paper_baseline == "v3_flat":
        return {
            "method_group": method.method_group,
            "policy": "v3_flat",
            "page_quant_scheme": method.page_quant_scheme,
            "key_bits": method.key_bits,
            "value_bits": method.value_bits,
            "residual_window": int(args.residual_window),
            "mixed_precision": False,
            "paper_baseline": "v3_flat",
            "integration": "turboquant/V3FlatCache+MSECompressor",
        }
    if method.paper_baseline in ("tq_pure", "tq_pure_mix"):
        from turboquant.block_cache.skvq_native_integration import (
            PAPER_CLIP,
            PAPER_SINK,
            PAPER_WINDOW,
            paper_pure_layer_protection,
        )

        prot_layers, prot_k, prot_v = paper_pure_layer_protection(
            method.paper_baseline, args
        )
        return {
            "block_size": args.block_size,
            "method_group": method.method_group,
            "policy": "hybrid",
            "quant_backend": method.quant_backend,
            "page_quant_scheme": method.page_quant_scheme,
            "sink": PAPER_SINK,
            "window": PAPER_WINDOW,
            "key_bits": method.key_bits,
            "value_bits": method.value_bits,
            "mixed_precision": method.mixed_precision,
            "important_ratio": args.important_ratio if method.mixed_precision else None,
            "importance_metric": method.importance_metric if method.mixed_precision else None,
            "high_key_bits": method.high_key_bits if method.mixed_precision else None,
            "high_value_bits": method.high_value_bits if method.mixed_precision else None,
            "low_key_bits": method.low_key_bits if method.mixed_precision else None,
            "low_value_bits": method.low_value_bits if method.mixed_precision else None,
            "paper_baseline": method.paper_baseline,
            "reorder": False,
            "protected_layers": prot_layers,
            "protected_key_bits": prot_k if method.paper_baseline == "tq_pure_mix" else None,
            "protected_value_bits": prot_v if method.paper_baseline == "tq_pure_mix" else None,
            "clipping": PAPER_CLIP,
            "quant_budget_per_update": getattr(args, "quant_budget_per_update", None),
            "integration": "turboquant-pytorch/BlockKVCache",
        }
    if method.paper_baseline == "skvq_native":
        from turboquant.block_cache.skvq_native_integration import skvq_native_config

        cfg = skvq_native_config(
            key_bits=method.key_bits,
            value_bits=method.value_bits,
            reorder_file=args.reorder_file,
        )
        cfg["method_group"] = method.method_group
        cfg["page_quant_scheme"] = method.page_quant_scheme
        cfg["quant_backend"] = method.quant_backend
        return cfg
    return {
        "block_size": args.block_size,
        "method_group": method.method_group,
        "sink": args.sink if method.policy == "hybrid" else None,
        "window": args.window if method.policy == "hybrid" else None,
        "policy": method.policy,
        "quant_backend": method.quant_backend,
        "page_quant_scheme": method.page_quant_scheme,
        "key_bits": method.key_bits,
        "value_bits": method.value_bits,
        "mixed_precision": method.mixed_precision,
        "importance_metric": method.importance_metric if method.mixed_precision else None,
        "important_ratio": args.important_ratio if method.mixed_precision else None,
        "high_key_bits": method.high_key_bits if method.mixed_precision else None,
        "high_value_bits": method.high_value_bits if method.mixed_precision else None,
        "low_key_bits": method.low_key_bits if method.mixed_precision else None,
        "low_value_bits": method.low_value_bits if method.mixed_precision else None,
        "num_layers": args.num_layers,
        "protected_layers": args.protected_layers,
        "protected_key_bits": args.protected_key_bits,
        "protected_value_bits": args.protected_value_bits,
        "group_size": args.group_size,
        "key_group_size": args.key_group_size,
        "value_group_size": args.value_group_size,
        "max_cached_decompressed_blocks": args.max_cached_decompressed_blocks,
        "incremental_materialize": getattr(args, "incremental_materialize", False),
        "quant_budget_per_update": getattr(args, "quant_budget_per_update", None),
    }
