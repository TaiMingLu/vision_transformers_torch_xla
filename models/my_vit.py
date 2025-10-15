# New timm VisionTransformer-based models with drop_path support
import copy
from timm.models.vision_transformer import (
    VisionTransformer,
    _create_vision_transformer,
    default_cfgs,
)
from ._registry import register_model

_CFG_LOOKUP = {
    'my_vit_mini': 'vit_tiny_patch16_224',
    'my_vit_ti': 'vit_tiny_patch16_224',
    'my_vit_xs': 'vit_small_patch16_224',
    'my_vit_s': 'vit_small_patch16_224',
    'my_vit_b': 'vit_base_patch16_224',
    'my_vit_l': 'vit_large_patch16_224',
}

for _model_name, _source_key in _CFG_LOOKUP.items():
    _src_cfg = default_cfgs.get(_model_name)
    if _src_cfg is None:
        _src_cfg = default_cfgs.get(_source_key)
    if _src_cfg is None:
        continue
    if hasattr(_src_cfg, 'to_dict'):
        default_cfgs[_model_name] = _src_cfg.to_dict()
    else:
        default_cfgs[_model_name] = copy.deepcopy(_src_cfg)


def _resolve_cfg(model_name: str):
    cfg = default_cfgs.get(model_name)
    if cfg is None:
        cfg_key = _CFG_LOOKUP.get(model_name)
        if cfg_key is not None:
            cfg = default_cfgs.get(cfg_key)
    if cfg is None:
        return None
    if hasattr(cfg, 'to_dict'):
        cfg = cfg.to_dict()
    return cfg


def _apply_default_cfg(model: VisionTransformer, model_name: str) -> VisionTransformer:
    cfg = _resolve_cfg(model_name)
    if not cfg:
        return model
    cfg_copy = copy.deepcopy(cfg)
    model.default_cfg = cfg_copy
    # Newer timm prefers pretrained_cfg attribute; fall back gracefully otherwise.
    try:
        model.pretrained_cfg = cfg_copy
    except AttributeError:
        pass
    return model

@register_model
def my_vit_mini(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Mini/16 with drop_path — ~3.3 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=144, depth=12, num_heads=3,
        # drop_path_rate=drop_path_by_size['mini']
    )
    model = _create_vision_transformer('my_vit_mini', pretrained=pretrained, **dict(model_args, **kwargs))
    return _apply_default_cfg(model, 'my_vit_mini')


@register_model
def my_vit_ti(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Tiny/16 with drop_path — 5.7 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        # drop_path_rate=drop_path_by_size['ti']
    )
    model = _create_vision_transformer('my_vit_ti', pretrained=pretrained, **dict(model_args, **kwargs))
    return _apply_default_cfg(model, 'my_vit_ti')


@register_model
def my_vit_xs(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-XS/16 with drop_path — ~11 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=288, depth=12, num_heads=4,
        # drop_path_rate=drop_path_by_size['xs']
    )
    model = _create_vision_transformer('my_vit_xs', pretrained=pretrained, **dict(model_args, **kwargs))
    return _apply_default_cfg(model, 'my_vit_xs')


@register_model
def my_vit_s(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Small/16 with drop_path — 22 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        # drop_path_rate=drop_path_by_size['s']
    )
    model = _create_vision_transformer('my_vit_s', pretrained=pretrained, **dict(model_args, **kwargs))
    return _apply_default_cfg(model, 'my_vit_s')


# @register_model
# def my_vit_m(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     """ ViT-Medium/16 with drop_path — 40 M params
#     """
#     model_args = dict(
#         patch_size=16, embed_dim=480, depth=12, num_heads=8,
#         # drop_path_rate=drop_path_by_size['m']
#     )
#     model = _create_vision_transformer('my_vit_m', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


@register_model
def my_vit_b(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Base/16 with drop_path — 86 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        # drop_path_rate=drop_path_by_size['b']
    )
    model = _create_vision_transformer('my_vit_b', pretrained=pretrained, **dict(model_args, **kwargs))
    return _apply_default_cfg(model, 'my_vit_b')


@register_model
def my_vit_l(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Large/16 with drop_path — 304 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        # drop_path_rate=drop_path_by_size['l']
    )
    model = _create_vision_transformer('my_vit_l', pretrained=pretrained, **dict(model_args, **kwargs))
    return _apply_default_cfg(model, 'my_vit_l')
