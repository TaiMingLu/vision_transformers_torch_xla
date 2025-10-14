# New timm VisionTransformer-based models with drop_path support
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer
from ._registry import register_model

@register_model
def my_vit_mini(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Mini/16 with drop_path — ~3.3 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=144, depth=12, num_heads=3,
        # drop_path_rate=drop_path_by_size['mini']
    )
    model = _create_vision_transformer('my_vit_mini', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def my_vit_ti(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Tiny/16 with drop_path — 5.7 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        # drop_path_rate=drop_path_by_size['ti']
    )
    model = _create_vision_transformer('my_vit_ti', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def my_vit_xs(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-XS/16 with drop_path — ~11 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=288, depth=12, num_heads=4,
        # drop_path_rate=drop_path_by_size['xs']
    )
    model = _create_vision_transformer('my_vit_xs', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def my_vit_s(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Small/16 with drop_path — 22 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        # drop_path_rate=drop_path_by_size['s']
    )
    model = _create_vision_transformer('my_vit_s', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


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
    return model


@register_model
def my_vit_l(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Large/16 with drop_path — 304 M params
    """
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        # drop_path_rate=drop_path_by_size['l']
    )
    model = _create_vision_transformer('my_vit_l', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

