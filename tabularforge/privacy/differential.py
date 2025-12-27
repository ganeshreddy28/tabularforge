"""
Differential Privacy Module
===========================

This module implements differential privacy mechanisms for synthetic data.

Differential privacy provides mathematical guarantees about privacy protection.
The key parameter is epsilon (ε):
    - Lower epsilon = stronger privacy, more noise, lower utility
    - Higher epsilon = weaker privacy, less noise, higher utility

Common epsilon values:
    - 0.1: Very strong privacy (significant noise)
    - 1.0: Balanced privacy/utility (commonly used)
    - 10.0: Weak privacy (minimal noise)

Author: Sai Ganesh Kolan
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================
from typing import Optional

import numpy as np
import pandas as pd


class DifferentialPrivacy:
    """
    Implements differential privacy for synthetic data.
    
    This class adds calibrated noise to synthetic data to provide formal
    privacy guarantees. The amount of noise is controlled by the epsilon
    parameter.
    
    Mechanism:
        We use the Laplace mechanism, which adds noise drawn from a
        Laplace distribution. The scale of the noise is inversely
        proportional to epsilon.
    
    Attributes:
        epsilon (float): Privacy budget (lower = more private)
        
    Example:
        >>> dp = DifferentialPrivacy(epsilon=1.0)
        >>> private_data = dp.add_noise(synthetic_data)
    """
    
    def __init__(self, epsilon: float = 1.0) -> None:
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget. Must be positive.
                - 0.1: Very strong privacy
                - 1.0: Balanced (default)
                - 10.0: Weak privacy
                
        Raises:
            ValueError: If epsilon is not positive
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        self.epsilon = epsilon
        
        # Calculate noise scale for Laplace mechanism
        # For normalized data with sensitivity ~1, scale = 1/epsilon
        self._noise_scale = 1.0 / epsilon
    
    def add_noise(
        self, 
        data: pd.DataFrame,
        sensitivity: float = 1.0
    ) -> pd.DataFrame:
        """
        Add differential privacy noise to the data.
        
        This method adds Laplace noise calibrated to provide epsilon-DP.
        The amount of noise depends on:
        1. Privacy budget (epsilon) - lower epsilon = more noise
        2. Sensitivity - how much a single record can affect the output
        
        Args:
            data: DataFrame to add noise to
            sensitivity: Maximum change a single record can cause.
                For normalized data, this is typically 1.0.
                
        Returns:
            DataFrame with added noise (copy, original unchanged)
        """
        # Make a copy to avoid modifying original
        noisy_data = data.copy()
        
        # Calculate noise scale based on sensitivity and epsilon
        # Laplace mechanism: scale = sensitivity / epsilon
        scale = sensitivity * self._noise_scale
        
        # Add noise to each numerical column
        for col in noisy_data.columns:
            # Only add noise to numerical columns
            if np.issubdtype(noisy_data[col].dtype, np.number):
                # Generate Laplace noise
                noise = np.random.laplace(
                    loc=0,
                    scale=scale,
                    size=len(noisy_data)
                )
                
                # Add noise to the column
                noisy_data[col] = noisy_data[col] + noise
        
        return noisy_data
    
    def get_privacy_guarantee(self) -> str:
        """
        Get a human-readable description of the privacy guarantee.
        
        Returns:
            String describing the privacy level
        """
        if self.epsilon <= 0.1:
            level = "Very Strong"
            description = "Significant noise added, may impact data utility"
        elif self.epsilon <= 1.0:
            level = "Strong"
            description = "Good balance of privacy and utility"
        elif self.epsilon <= 5.0:
            level = "Moderate"
            description = "Moderate privacy protection"
        else:
            level = "Weak"
            description = "Minimal noise, lower privacy protection"
        
        return (
            f"Privacy Level: {level}\n"
            f"Epsilon: {self.epsilon}\n"
            f"Description: {description}"
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DifferentialPrivacy(epsilon={self.epsilon})"
