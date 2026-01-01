"""
Privacy Module

This module contains privacy-related functionality for TabularForge.

Components:
    - DifferentialPrivacy: Adds calibrated noise for formal privacy guarantees
    - PrivacyAttackSimulator: Simulates attacks to evaluate privacy protection

Usage:
    >>> from tabularforge.privacy import DifferentialPrivacy
    >>> dp = DifferentialPrivacy(epsilon=1.0)
    >>> private_data = dp.add_noise(synthetic_data)
"""

from tabularforge.privacy.differential import DifferentialPrivacy
from tabularforge.privacy.attacks import PrivacyAttackSimulator

__all__ = [
    "DifferentialPrivacy",
    "PrivacyAttackSimulator",
]
