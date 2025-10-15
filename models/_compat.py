"""Helpers for bridging timm API differences across versions."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Iterable, Mapping


def _resolve_symbol(name: str, candidates: Iterable[str], install_hint: str) -> object:
    """Return the first matching attribute from the candidate module list."""

    for module_name in candidates:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        symbol = getattr(module, name, None)
        if symbol is not None:
            return symbol

    raise ImportError(
        f"{name} is not available in the installed timm package. Install a release that includes it, "
        f"for example {install_hint}."
    )


_SYMBOL_SOURCES: Mapping[str, Iterable[str]] = {
    "Attention": (
        "timm.layers",
        "timm.layers.attention",
        "timm.layers.attention_pool2d",
        "timm.models.layers",
        "timm.models.layers.attention",
        "timm.models.vision_transformer",
    ),
    "AttentionPoolLatent": (
        "timm.layers",
        "timm.layers.attention_pool2d",
        "timm.models.layers",
        "timm.models.layers.attention",
        "timm.models.vision_transformer",
    ),
    "PatchEmbed": (
        "timm.layers",
        "timm.layers.patch_embed",
        "timm.models.layers",
        "timm.models.layers.patch_embed",
        "timm.models.vision_transformer",
    ),
    "Mlp": (
        "timm.layers",
        "timm.layers.mlp",
        "timm.layers.activations",
        "timm.models.layers",
        "timm.models.layers.mlp",
        "timm.models.vision_transformer",
    ),
    "SwiGLUPacked": (
        "timm.layers",
        "timm.layers.mlp",
        "timm.layers.activations",
        "timm.models.layers",
        "timm.models.layers.mlp",
    ),
    "SwiGLU": (
        "timm.layers",
        "timm.layers.mlp",
        "timm.layers.activations",
        "timm.models.layers",
        "timm.models.layers.mlp",
    ),
    "LayerNorm": (
        "timm.layers",
        "timm.layers.norm",
        "timm.layers.norm_act",
        "timm.models.layers",
        "timm.models.layers.norm",
    ),
    "RmsNorm": (
        "timm.layers",
        "timm.layers.norm",
        "timm.models.layers",
        "timm.models.layers.norm",
    ),
    "DropPath": (
        "timm.layers",
        "timm.layers.drop",
        "timm.models.layers",
        "timm.models.layers.drop",
    ),
    "PatchDropout": (
        "timm.layers",
        "timm.layers.drop",
        "timm.models.layers",
        "timm.models.layers.drop",
        "timm.models.vision_transformer",
    ),
    "trunc_normal_": (
        "timm.layers",
        "timm.layers.weight_init",
        "timm.models.layers",
        "timm.models.layers.weight_init",
    ),
    "lecun_normal_": (
        "timm.layers",
        "timm.layers.weight_init",
        "timm.models.layers",
        "timm.models.layers.weight_init",
    ),
    "resample_patch_embed": (
        "timm.layers",
        "timm.layers.pos_embed",
        "timm.models.layers",
        "timm.models.layers.pos_embed",
        "timm.models.vision_transformer",
    ),
    "resample_abs_pos_embed": (
        "timm.layers",
        "timm.layers.pos_embed",
        "timm.models.layers",
        "timm.models.layers.pos_embed",
        "timm.models.vision_transformer",
    ),
    "resample_abs_pos_embed_nhwc": (
        "timm.layers",
        "timm.layers.pos_embed",
        "timm.models.layers",
        "timm.models.layers.pos_embed",
    ),
    "use_fused_attn": (
        "timm.layers",
        "timm.layers.attention",
        "timm.layers.attention_ops",
        "timm.models.layers",
        "timm.models.layers.attention",
    ),
    "get_act_layer": (
        "timm.layers",
        "timm.layers.create_act",
        "timm.models.layers",
        "timm.models.layers.create_act",
    ),
    "get_norm_layer": (
        "timm.layers",
        "timm.layers.create_norm",
        "timm.models.layers",
        "timm.models.layers.create_norm",
    ),
    "LayerType": (
        "timm.layers",
        "timm.layers.norm",
        "timm.models.layers",
        "timm.models.layers.norm",
    ),
    "HybridEmbed": (
        "timm.layers",
        "timm.layers.hybrid",
        "timm.layers.hybrid_embed",
        "timm.models.layers",
        "timm.models.layers.hybrid_embed",
        "timm.models.vision_transformer",
    ),
    "maybe_add_mask": (
        "timm.layers.helpers",
        "timm.layers.attention",
        "timm.layers.attention_pool2d",
        "timm.layers.pos_embed",
        "timm.models.layers.helpers",
        "timm.models.layers.attention",
        "timm.models.layers.pos_embed",
    ),
}


def _fallback_maybe_add_mask() -> Callable:
    import torch

    def maybe_add_mask(attn, attn_mask=None):
        if attn_mask is None:
            return attn
        if torch.is_tensor(attn_mask) and attn_mask.dtype == torch.bool:
            return attn.masked_fill(~attn_mask, float('-inf'))
        return attn + attn_mask

    return maybe_add_mask


_FALLBACK_FACTORIES: Dict[str, Callable[[], object]] = {
    "maybe_add_mask": _fallback_maybe_add_mask,
}


def _load_symbols() -> Dict[str, object]:
    namespace: Dict[str, object] = {}
    for name, modules in _SYMBOL_SOURCES.items():
        try:
            namespace[name] = _resolve_symbol(name, modules, "'timm==0.9.16'")
        except ImportError:
            factory = _FALLBACK_FACTORIES.get(name)
            if factory is None:
                raise
            namespace[name] = factory()
    return namespace


_EXPORTS = _load_symbols()

__all__ = list(_EXPORTS.keys())

globals().update(_EXPORTS)
