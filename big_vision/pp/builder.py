"""Minimal Big Vision preprocessing pipeline builder.

This module implements just enough of the original Big Vision builder to keep
our TFDS ImageNet loader unchanged. Pipelines are expressed as strings such as
``"decode|resize_small(256)|central_crop(224)|value_range(0, 1)|keep('image','label')"``.

Each operation is registered via :func:`register`. The builder parses the string
into a list of callables and returns a function that applies them sequentially
to TFDS examples.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Dict, Iterable, List, Sequence

import tensorflow as tf

Operation = Callable[[Dict[str, Any]], Dict[str, Any]]
OperationFactory = Callable[..., Operation]


class PreprocessRegistry:
    """Keeps track of preprocessing operations by name."""

    def __init__(self) -> None:
        self._ops: Dict[str, OperationFactory] = {}

    def register(self, name: str, factory: OperationFactory) -> None:
        if name in self._ops:
            raise ValueError(f"Operation '{name}' already registered")
        self._ops[name] = factory

    def get(self, name: str) -> OperationFactory:
        try:
            return self._ops[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._ops))
            raise KeyError(f"Unknown preprocessing op '{name}'. Available ops: {available}") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._ops


_REGISTRY = PreprocessRegistry()


def register(name: str) -> Callable[[OperationFactory], OperationFactory]:
    """Decorator used by ops modules to register an operation."""

    def decorator(factory: OperationFactory) -> OperationFactory:
        _REGISTRY.register(name, factory)
        return factory

    return decorator


def _instantiate(op_spec: str) -> Operation:
    """Instantiate a registered preprocessing op from a spec string."""

    op_spec = op_spec.strip()
    if not op_spec:
        raise ValueError("Empty operation spec")

    if "(" not in op_spec:
        factory = _REGISTRY.get(op_spec)
        return factory()

    node = ast.parse(op_spec, mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Unsupported operation spec '{op_spec}'")
    if not isinstance(node.func, ast.Name):
        raise ValueError(f"Unsupported operation target in '{op_spec}'")

    name = node.func.id
    factory = _REGISTRY.get(name)

    args = [ast.literal_eval(arg) for arg in node.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return factory(*args, **kwargs)


def _parse_pipeline(spec: Sequence[str] | str) -> List[Operation]:
    if isinstance(spec, str):
        steps = [step.strip() for step in spec.split("|") if step.strip()]
    else:
        steps = []
        for element in spec:
            steps.extend(_parse_pipeline(element))

    if not steps:
        raise ValueError("Preprocess pipeline spec is empty")

    return [_instantiate(step) for step in steps]


def get_preprocess_fn(spec: Sequence[str] | str,
                      log_data: bool = False,
                      log_steps: bool = False) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a preprocessing function matching the given pipeline specification."""

    operations = _parse_pipeline(spec)

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        data = example
        for op in operations:
            data = op(data)
        if log_data:
            tf.print("[big_vision] example keys:", sorted(list(data.keys())))
        if log_steps:
            tf.print("[big_vision] applied", len(operations), "ops")
        return data

    return preprocess


# Import ops modules for side effects so they register themselves.
from . import ops_general as _ops_general  # noqa: E402,F401
from . import ops_image as _ops_image  # noqa: E402,F401

__all__ = ["get_preprocess_fn", "register"]
