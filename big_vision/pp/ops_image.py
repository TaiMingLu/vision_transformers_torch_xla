"""Image preprocessing ops for the vendored Big Vision subset."""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from .builder import register


def _copy(example: Dict[str, Any]) -> Dict[str, Any]:
    return dict(example)


def _ensure_3d(image: tf.Tensor, channels: int) -> tf.Tensor:
    image.set_shape([None, None, channels])
    return image


@register("decode")
def decode(key: str = "image", channels: int = 3):
    """Decode an encoded image string into a float32 tensor in [0, 1]."""

    def _op(example: Dict[str, Any]) -> Dict[str, Any]:
        data = _copy(example)
        image = tf.image.decode_image(data[key], channels=channels, expand_animations=False)
        image = _ensure_3d(image, channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        data[key] = image
        return data

    return _op


@register("decode_jpeg_and_inception_crop")
def decode_jpeg_and_inception_crop(size: int,
                                   area_min: float = 0.08,
                                   area_max: float = 1.0,
                                   aspect_ratio_min: float = 0.75,
                                   aspect_ratio_max: float = 1.3333333,
                                   key: str = "image",
                                   channels: int = 3):
    """Decode JPEG bytes and apply Inception-style random resized cropping."""

    def _op(example: Dict[str, Any]) -> Dict[str, Any]:
        data = _copy(example)
        image = tf.io.decode_jpeg(data[key], channels=channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = _ensure_3d(image, channels)

        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.zeros([0, 0, 4], tf.float32),
            min_object_covered=0.0,
            aspect_ratio_range=[aspect_ratio_min, aspect_ratio_max],
            area_range=[area_min, area_max],
            max_attempts=10,
            use_image_if_no_bounding_boxes=True,
        )
        image = tf.slice(image, bbox_begin, bbox_size)
        image = tf.image.resize(image, [size, size], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        data[key] = image
        return data

    return _op


@register("flip_lr")
def flip_lr(key: str = "image"):
    """Randomly flip the image horizontally with probability 0.5."""

    def _op(example: Dict[str, Any]) -> Dict[str, Any]:
        data = _copy(example)
        image = tf.convert_to_tensor(data[key])
        image = tf.cond(tf.random.uniform([], 0.0, 1.0) < 0.5,
                        lambda: tf.image.flip_left_right(image),
                        lambda: image)
        data[key] = image
        return data

    return _op


@register("resize_small")
def resize_small(size: int, key: str = "image", method: str = "bicubic"):
    """Resize the image so the shorter edge equals ``size`` while preserving aspect ratio."""

    resize_methods = {
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "lanczos3": tf.image.ResizeMethod.LANCZOS3,
        "lanczos5": tf.image.ResizeMethod.LANCZOS5,
        "gaussian": tf.image.ResizeMethod.GAUSSIAN,
        "mitchellcubic": tf.image.ResizeMethod.MITCHELLCUBIC,
    }
    if method not in resize_methods:
        raise ValueError(f"Unsupported resize method '{method}'")

    resize_method = resize_methods[method]

    def _op(example: Dict[str, Any]) -> Dict[str, Any]:
        data = _copy(example)
        image = tf.convert_to_tensor(data[key])
        if image.dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        height, width = shape[0], shape[1]
        min_dim = tf.minimum(height, width)
        scale = tf.cast(size, tf.float32) / tf.maximum(1.0, min_dim)
        new_height = tf.cast(tf.round(height * scale), tf.int32)
        new_width = tf.cast(tf.round(width * scale), tf.int32)
        image = tf.image.resize(image, [new_height, new_width], method=resize_method, antialias=True)
        data[key] = image
        return data

    return _op


@register("central_crop")
def central_crop(size: int, key: str = "image"):
    """Center-crop or pad the image to a ``size``×``size`` square."""

    def _op(example: Dict[str, Any]) -> Dict[str, Any]:
        data = _copy(example)
        image = tf.convert_to_tensor(data[key])
        if image.dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, size, size)
        data[key] = image
        return data

    return _op
