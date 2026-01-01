"""
Metrics Module

This module contains metrics for evaluating synthetic data quality and privacy.

Components:
    - StatisticalMetrics: Measures statistical similarity
    - UtilityMetrics: Measures ML utility
    - PrivacyMetrics: Measures privacy protection

Usage:
    >>> from tabularforge.metrics import StatisticalMetrics
    >>> metrics = StatisticalMetrics()
    >>> scores = metrics.compute(real_data, synthetic_data, encoder)
"""

from tabularforge.metrics.statistical import StatisticalMetrics
from tabularforge.metrics.utility import UtilityMetrics
from tabularforge.metrics.privacy import PrivacyMetrics

__all__ = [
    "StatisticalMetrics",
    "UtilityMetrics",
    "PrivacyMetrics",
]
