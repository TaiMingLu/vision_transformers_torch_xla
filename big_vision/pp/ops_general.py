"""General-purpose preprocessing ops compatible with the Big Vision API."""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from .builder import register


@register("keep")
def keep(*keys: str):
    """Keep only the requested keys from the example dictionary."""

    if not keys:
        raise ValueError("keep() requires at least one key name")
    normalized = tuple(str(k) for k in keys)

    def _op(example: Dict[str, Any]) -> Dict[str, Any]:
        return {k: example[k] for k in normalized}

    return _op


@register("value_range")
def value_range(min_value: float, max_value: float, key: str = "image"):
    """Re-scale image values from [0, 255] (or [0, 1]) to the requested range."""

    scale = max_value - min_value

    def _op(example: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(example)
        image = tf.convert_to_tensor(data[key])
        if image.dtype.is_integer:
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.cast(image, tf.float32)
        image = image * scale + min_value
        data[key] = image
        return data

    return _op
