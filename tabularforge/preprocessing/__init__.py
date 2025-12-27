"""
Preprocessing Module
====================

This module contains data preprocessing utilities for TabularForge.

Components:
    - DataEncoder: Handles column type detection and encoding/decoding
    - DataTransformer: Orchestrates the full transformation pipeline

Usage:
    >>> from tabularforge.preprocessing import DataEncoder, DataTransformer
    >>> encoder = DataEncoder()
    >>> encoder.fit(data)
    >>> transformer = DataTransformer()
    >>> transformed = transformer.fit_transform(data, encoder)
"""

from tabularforge.preprocessing.encoder import DataEncoder
from tabularforge.preprocessing.transformer import DataTransformer

__all__ = [
    "DataEncoder",
    "DataTransformer",
]
