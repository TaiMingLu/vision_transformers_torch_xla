"""Helpers for bridging timm API differences across versions."""

from __future__ import annotations

from importlib import import_module
from typing import Iterable


def _resolve_hybrid_embed() -> type:
    candidates: Iterable[str] = (
        "timm.layers",
        "timm.layers.hybrid",
        "timm.layers.hybrid_embed",
        "timm.models.layers",
        "timm.models.layers.hybrid_embed",
        "timm.models.vision_transformer",
    )

    for module_name in candidates:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        hybrid = getattr(module, "HybridEmbed", None)
        if hybrid is not None:
            return hybrid

    raise ImportError(
        "HybridEmbed is not available in the installed timm package. "
        "Install a timm release that includes vision transformer layers, e.g. 'timm==0.9.16'."
    )


HybridEmbed = _resolve_hybrid_embed()

__all__ = ["HybridEmbed"]
