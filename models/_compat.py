"""Helpers for bridging timm API differences across versions."""

from __future__ import annotations

try:
    # timm <=0.9 exposes HybridEmbed from timm.layers
    from timm.layers import HybridEmbed  # noqa: F401
except ImportError:  # pragma: no cover - triggered on newer timm builds
    # timm >=0.10 moved HybridEmbed under timm.models.layers
    from timm.models.layers import HybridEmbed  # noqa: F401

__all__ = ["HybridEmbed"]
