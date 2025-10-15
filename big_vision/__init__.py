"""Lightweight in-tree replacement for the Big Vision package.

This module provides the subset of the Big Vision preprocessing pipeline
required by the local TFDS ImageNet loader. It exposes the same public
interfaces used in datasets.py so downstream code can remain unchanged.
"""

from . import pp  # re-export package for compatibility

__all__ = ["pp"]
