"""
Statistical Metrics Module

This module provides metrics for evaluating how well synthetic data
matches the statistical properties of real data.

Metrics include:
    - Distribution similarity (per column)
    - Correlation preservation
    - Summary statistics comparison

Author: Sai Ganesh Kolan
License: MIT
"""

# IMPORTS
# =============================================================================
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats


class StatisticalMetrics:
    """
    Computes statistical similarity metrics between real and synthetic data.
    
    These metrics evaluate how well the synthetic data preserves the
    statistical properties of the original data.
    
    Metrics:
        - Distribution: How similar are the distributions of each column?
        - Correlation: How well are column correlations preserved?
        - Summary Stats: How close are mean, std, min, max?
    
    Example:
        >>> metrics = StatisticalMetrics()
        >>> scores = metrics.compute(real_data, synthetic_data, encoder)
        >>> print(f"Overall similarity: {scores['overall']:.2%}")
    """
    
    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        encoder: Any
    ) -> Dict[str, float]:
        """
        Compute all statistical metrics.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            encoder: DataEncoder with column type information
            
        Returns:
            Dictionary with metric scores (0-1, higher is better)
        """
        scores = {}
        
        # Distribution similarity
        scores["distribution"] = self._distribution_similarity(
            real_data, synthetic_data, encoder
        )
        
        # Correlation preservation
        scores["correlation"] = self._correlation_similarity(
            real_data, synthetic_data, encoder
        )
        
        # Summary statistics
        scores["summary_stats"] = self._summary_stats_similarity(
            real_data, synthetic_data, encoder
        )
        
        # Overall score (weighted average)
        scores["overall"] = (
            0.4 * scores["distribution"] +
            0.4 * scores["correlation"] +
            0.2 * scores["summary_stats"]
        )
        
        return scores
    
    def _distribution_similarity(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        encoder: Any
    ) -> float:
        """
        Calculate distribution similarity using KS test for numerical
        and chi-squared test for categorical columns.
        
        Returns:
            Similarity score (0-1)
        """
        similarities = []
        
        # Numerical columns - use Kolmogorov-Smirnov test
        for col in encoder.numerical_columns:
            if col in real_data.columns and col in synthetic_data.columns:
                real_col = real_data[col].dropna()
                synth_col = synthetic_data[col].dropna()
                
                if len(real_col) > 0 and len(synth_col) > 0:
                    # KS statistic (0 = identical, 1 = completely different)
                    ks_stat, _ = stats.ks_2samp(real_col, synth_col)
                    # Convert to similarity (1 = identical)
                    similarity = 1 - ks_stat
                    similarities.append(similarity)
        
        # Categorical columns - use frequency comparison
        for col in encoder.categorical_columns:
            if col in real_data.columns and col in synthetic_data.columns:
                real_counts = real_data[col].value_counts(normalize=True)
                synth_counts = synthetic_data[col].value_counts(normalize=True)
                
                # Align indices
                all_categories = set(real_counts.index) | set(synth_counts.index)
                real_freq = np.array([real_counts.get(c, 0) for c in all_categories])
                synth_freq = np.array([synth_counts.get(c, 0) for c in all_categories])
                
                # Total Variation Distance
                tvd = 0.5 * np.sum(np.abs(real_freq - synth_freq))
                similarity = 1 - tvd
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _correlation_similarity(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        encoder: Any
    ) -> float:
        """
        Calculate how well column correlations are preserved.
        
        Returns:
            Similarity score (0-1)
        """
        # Get numerical columns only
        num_cols = [
            col for col in encoder.numerical_columns
            if col in real_data.columns and col in synthetic_data.columns
        ]
        
        if len(num_cols) < 2:
            return 1.0  # Can't compute correlation with < 2 columns
        
        # Compute correlation matrices
        real_corr = real_data[num_cols].corr().values
        synth_corr = synthetic_data[num_cols].corr().values
        
        # Handle NaN
        real_corr = np.nan_to_num(real_corr, nan=0)
        synth_corr = np.nan_to_num(synth_corr, nan=0)
        
        # Frobenius norm of difference
        diff = np.linalg.norm(real_corr - synth_corr, 'fro')
        max_diff = np.sqrt(2 * len(num_cols) ** 2)  # Maximum possible difference
        
        similarity = 1 - (diff / max_diff)
        return max(0, similarity)
    
    def _summary_stats_similarity(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        encoder: Any
    ) -> float:
        """
        Compare summary statistics (mean, std) between real and synthetic.
        
        Returns:
            Similarity score (0-1)
        """
        similarities = []
        
        for col in encoder.numerical_columns:
            if col in real_data.columns and col in synthetic_data.columns:
                real_col = real_data[col].dropna()
                synth_col = synthetic_data[col].dropna()
                
                if len(real_col) > 0 and len(synth_col) > 0:
                    # Mean comparison
                    real_mean = real_col.mean()
                    synth_mean = synth_col.mean()
                    real_std = real_col.std() + 1e-6
                    
                    mean_diff = abs(real_mean - synth_mean) / real_std
                    mean_sim = np.exp(-mean_diff)  # Exponential decay
                    
                    # Std comparison
                    synth_std = synth_col.std() + 1e-6
                    std_ratio = min(real_std, synth_std) / max(real_std, synth_std)
                    
                    # Combined similarity
                    similarity = (mean_sim + std_ratio) / 2
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def __repr__(self) -> str:
        return "StatisticalMetrics()"
