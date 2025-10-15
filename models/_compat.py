"""Helpers for bridging timm API differences across versions."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union


def _resolve_hybrid_embed():
    """Return timm's HybridEmbed regardless of release layout."""

    try:
        from timm.layers import HybridEmbed as _HybridEmbed  # noqa: F401

        return _HybridEmbed
    except ImportError:
        pass

    try:
        from timm.models.layers import HybridEmbed as _HybridEmbed  # noqa: F401

        return _HybridEmbed
    except ImportError:
        pass

    try:
        from timm.models.vision_transformer import HybridEmbed as _HybridEmbed  # noqa: F401

        return _HybridEmbed
    except ImportError:
        pass

    return None


_HybridEmbed = _resolve_hybrid_embed()


if _HybridEmbed is None:
    import torch
    import torch.nn as nn

    Size2D = Union[int, Sequence[int], Tuple[int, int]]

    def _to_2tuple(value: Size2D) -> Tuple[int, int]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, Sequence):
            value = list(value)
            if len(value) != 2:
                raise ValueError(f"Expected sequence of length 2, got {value!r}")
            return int(value[0]), int(value[1])
        value = int(value)
        return value, value

    class HybridEmbed(nn.Module):
        """Lightweight fallback for timm HybridEmbed."""

        def __init__(
                self,
                backbone: nn.Module,
                img_size: Size2D = 224,
                feature_size: Optional[Size2D] = None,
                in_chans: int = 3,
                embed_dim: int = 768,
                proj: bool = True,
                flatten: bool = True,
                bias: bool = True,
                **_: object,
        ) -> None:
            super().__init__()
            if backbone is None:
                raise ValueError("backbone must be a Module when using HybridEmbed fallback")
            self.backbone = backbone
            self.flatten = flatten

            img_size = _to_2tuple(img_size)
            feature_size_tuple: Optional[Tuple[int, int]]
            feature_dim: Optional[int]

            training = backbone.training
            backbone.eval()

            with torch.no_grad():
                if feature_size is None:
                    device = torch.device("cpu")
                    try:
                        device = next(backbone.parameters()).device
                    except StopIteration:
                        pass
                    dummy = torch.zeros(1, in_chans, img_size[0], img_size[1], device=device)
                    output = backbone(dummy)
                    if isinstance(output, (list, tuple)):
                        output = output[-1]
                    feature_dim = output.shape[1]
                    feature_size_tuple = (output.shape[2], output.shape[3])
                else:
                    feature_size_tuple = _to_2tuple(feature_size)
                    feature_dim = None
                    if hasattr(backbone, 'feature_info'):
                        try:
                            feature_dim = backbone.feature_info.channels()[-1]
                        except Exception:  # pragma: no cover - optional API
                            feature_info = getattr(backbone.feature_info, '__getitem__', None)
                            if feature_info is not None:
                                try:
                                    feature_dim = feature_info[-1]['num_chs']
                                except Exception:
                                    pass
                if feature_dim is None:
                    # last-resort forward to infer channels even if feature_size supplied
                    device = torch.device("cpu")
                    try:
                        device = next(backbone.parameters()).device
                    except StopIteration:
                        pass
                    dummy = torch.zeros(1, in_chans, img_size[0], img_size[1], device=device)
                    output = backbone(dummy)
                    if isinstance(output, (list, tuple)):
                        output = output[-1]
                    feature_dim = output.shape[1]
                    if feature_size_tuple is None:
                        feature_size_tuple = (output.shape[2], output.shape[3])

            if training:
                backbone.train()

            if feature_size_tuple is None or feature_dim is None:
                raise RuntimeError("Unable to infer hybrid embedding shape from backbone output")

            self.num_patches = feature_size_tuple[0] * feature_size_tuple[1]
            self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1, bias=bias) if proj else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            x = self.backbone(x)
            if isinstance(x, (list, tuple)):
                x = x[-1]
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)
            return x

else:
    HybridEmbed = _HybridEmbed


__all__ = ["HybridEmbed"]
