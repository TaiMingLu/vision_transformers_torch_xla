"""Helpers for bridging timm API differences across versions."""

from __future__ import annotations


def _resolve_hybrid_embed():
    """Return timm's HybridEmbed regardless of release layout."""

    # timm <=0.9 exposes HybridEmbed from timm.layers
    try:
        from timm.layers import HybridEmbed as _HybridEmbed  # noqa: F401

        return _HybridEmbed
    except ImportError:
        pass

    # timm >=0.10 moved HybridEmbed under timm.models.layers
    try:
        from timm.models.layers import HybridEmbed as _HybridEmbed  # noqa: F401

        return _HybridEmbed
    except ImportError:
        pass

    # some builds only keep the class inside vision_transformer
    try:
        from timm.models.vision_transformer import HybridEmbed as _HybridEmbed  # noqa: F401

        return _HybridEmbed
    except ImportError as err:
        raise ImportError(
            "HybridEmbed not found in timm; confirm timm installation includes vision transformer ops"
        ) from err


# cache the resolved class to avoid redundant imports at runtime
HybridEmbed = _resolve_hybrid_embed()

__all__ = ["HybridEmbed"]
