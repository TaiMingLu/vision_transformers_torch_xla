"""Helpers for bridging timm API differences across versions."""

from __future__ import annotations

from importlib import import_module
from typing import Iterable


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


HybridEmbed = _resolve_symbol(
    "HybridEmbed",
    (
        "timm.layers",
        "timm.layers.hybrid",
        "timm.layers.hybrid_embed",
        "timm.models.layers",
        "timm.models.layers.hybrid_embed",
        "timm.models.vision_transformer",
    ),
    "'timm==0.9.16'",
)

maybe_add_mask = _resolve_symbol(
    "maybe_add_mask",
    (
        "timm.layers",
        "timm.layers.attention",
        "timm.layers.attention_pool2d",
        "timm.layers.pos_embed",
        "timm.layers.helpers",
        "timm.models.layers",
        "timm.models.layers.attention",
        "timm.models.vision_transformer",
    ),
    "'timm==0.9.16'",
)


__all__ = ["HybridEmbed", "maybe_add_mask"]
