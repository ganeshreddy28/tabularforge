"""
Generators Module

This module contains all synthetic data generators available in TabularForge.

Available Generators:
    - GaussianCopulaGenerator: Fast, correlation-preserving generator
    - CTGANGenerator: Deep learning generator using GANs
    - TVAEGenerator: Deep learning generator using VAEs

Usage:
    >>> from tabularforge.generators import GaussianCopulaGenerator
    >>> generator = GaussianCopulaGenerator()
    >>> generator.fit(data, encoder)
    >>> synthetic = generator.sample(1000)
"""

from tabularforge.generators.base import BaseGenerator
from tabularforge.generators.copula import GaussianCopulaGenerator
from tabularforge.generators.ctgan import CTGANGenerator
from tabularforge.generators.tvae import TVAEGenerator

__all__ = [
    "BaseGenerator",
    "GaussianCopulaGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
]
