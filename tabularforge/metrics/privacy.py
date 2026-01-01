"""
Privacy Metrics Module

This module provides metrics for evaluating privacy protection
of synthetic data.

Metrics include:
    - Membership inference risk
    - Attribute inference risk
    - Nearest neighbor distance

Author: Sai Ganesh Kolan
License: MIT
"""

# IMPORTS
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from tabularforge.privacy.attacks import PrivacyAttackSimulator


class PrivacyMetrics:
    """
    Computes privacy metrics for synthetic data.
    
    These metrics evaluate how well the synthetic data protects
    individual privacy. Lower risk scores indicate better privacy.
    
    Example:
        >>> metrics = PrivacyMetrics()
        >>> scores = metrics.compute(real_data, synthetic_data, encoder)
        >>> print(f"Membership inference risk: {scores['membership_inference_risk']:.2%}")
    """
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize privacy metrics calculator.
        
        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        self._simulator = PrivacyAttackSimulator(random_state=random_state)
    
    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        encoder: Any
    ) -> Dict[str, float]:
        """
        Compute all privacy metrics.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            encoder: DataEncoder with column type information
            
        Returns:
            Dictionary with privacy metrics
        """
        # Run all attacks
        attack_results = self._simulator.run_all_attacks(
            real_data, synthetic_data
        )
        
        return attack_results
    
    def __repr__(self) -> str:
        return f"PrivacyMetrics(random_state={self.random_state})"
