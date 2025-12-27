"""
Utility Metrics Module
======================

This module provides metrics for evaluating the ML utility of synthetic data.

ML utility measures how well models trained on synthetic data perform
compared to models trained on real data. High utility means synthetic
data is a good substitute for real data in ML workflows.

Author: Sai Ganesh Kolan
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score


class UtilityMetrics:
    """
    Computes ML utility metrics for synthetic data.
    
    ML utility measures how useful synthetic data is for training ML models.
    The key test is: Train on Synthetic, Test on Real (TSTR).
    
    If a model trained on synthetic data performs similarly to one trained
    on real data, the synthetic data has high utility.
    
    Example:
        >>> metrics = UtilityMetrics()
        >>> scores = metrics.compute(real_data, synthetic_data, encoder)
        >>> print(f"ML utility: {scores['overall']:.2%}")
    """
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize utility metrics calculator.
        
        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
    
    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        encoder: Any
    ) -> Dict[str, float]:
        """
        Compute ML utility metrics.
        
        This runs the TSTR (Train on Synthetic, Test on Real) evaluation
        for multiple target columns.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            encoder: DataEncoder with column type information
            
        Returns:
            Dictionary with utility scores
        """
        scores = {}
        
        # Test with different target columns
        tstr_scores = []
        
        # Try categorical columns as targets (classification)
        for target_col in encoder.categorical_columns[:3]:  # Limit to 3
            if target_col in real_data.columns and target_col in synthetic_data.columns:
                score = self._tstr_classification(
                    real_data, synthetic_data, target_col
                )
                if score is not None:
                    tstr_scores.append(score)
        
        # Try numerical columns as targets (regression)
        for target_col in encoder.numerical_columns[:3]:  # Limit to 3
            if target_col in real_data.columns and target_col in synthetic_data.columns:
                score = self._tstr_regression(
                    real_data, synthetic_data, target_col
                )
                if score is not None:
                    tstr_scores.append(score)
        
        # Overall utility score
        scores["tstr_scores"] = tstr_scores
        scores["overall"] = np.mean(tstr_scores) if tstr_scores else 0.5
        
        return scores
    
    def _tstr_classification(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_col: str
    ) -> Optional[float]:
        """
        Train on Synthetic, Test on Real for classification.
        
        Compares:
        - Model trained on real data, tested on real data (baseline)
        - Model trained on synthetic data, tested on real data (TSTR)
        
        Returns:
            Ratio of TSTR accuracy to baseline accuracy (0-1+)
        """
        try:
            # Prepare features and target
            feature_cols = [c for c in real_data.columns if c != target_col]
            
            X_real = real_data[feature_cols].copy()
            y_real = real_data[target_col].copy()
            X_synth = synthetic_data[feature_cols].copy()
            y_synth = synthetic_data[target_col].copy()
            
            # Convert to numeric
            for col in feature_cols:
                if X_real[col].dtype == object:
                    X_real[col] = pd.Categorical(X_real[col]).codes
                    X_synth[col] = pd.Categorical(X_synth[col]).codes
            
            if y_real.dtype == object:
                y_real = pd.Categorical(y_real).codes
                y_synth = pd.Categorical(y_synth).codes
            
            # Fill NaN
            X_real = X_real.fillna(0)
            X_synth = X_synth.fillna(0)
            
            # Split real data
            X_train, X_test, y_train, y_test = train_test_split(
                X_real, y_real, test_size=0.3, random_state=self.random_state
            )
            
            # Baseline: Train on real, test on real
            clf_real = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )
            clf_real.fit(X_train, y_train)
            baseline_acc = accuracy_score(y_test, clf_real.predict(X_test))
            
            # TSTR: Train on synthetic, test on real
            clf_synth = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )
            clf_synth.fit(X_synth, y_synth)
            tstr_acc = accuracy_score(y_test, clf_synth.predict(X_test))
            
            # Ratio (capped at 1.0)
            if baseline_acc > 0:
                return min(tstr_acc / baseline_acc, 1.0)
            return 0.5
            
        except Exception:
            return None
    
    def _tstr_regression(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_col: str
    ) -> Optional[float]:
        """
        Train on Synthetic, Test on Real for regression.
        
        Returns:
            Ratio of TSTR R² to baseline R² (0-1+)
        """
        try:
            # Prepare features and target
            feature_cols = [c for c in real_data.columns if c != target_col]
            
            X_real = real_data[feature_cols].copy()
            y_real = real_data[target_col].copy()
            X_synth = synthetic_data[feature_cols].copy()
            y_synth = synthetic_data[target_col].copy()
            
            # Convert to numeric
            for col in feature_cols:
                if X_real[col].dtype == object:
                    X_real[col] = pd.Categorical(X_real[col]).codes
                    X_synth[col] = pd.Categorical(X_synth[col]).codes
            
            # Fill NaN
            X_real = X_real.fillna(0)
            X_synth = X_synth.fillna(0)
            y_real = y_real.fillna(y_real.mean())
            y_synth = y_synth.fillna(y_synth.mean())
            
            # Split real data
            X_train, X_test, y_train, y_test = train_test_split(
                X_real, y_real, test_size=0.3, random_state=self.random_state
            )
            
            # Baseline: Train on real, test on real
            reg_real = RandomForestRegressor(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )
            reg_real.fit(X_train, y_train)
            baseline_r2 = r2_score(y_test, reg_real.predict(X_test))
            
            # TSTR: Train on synthetic, test on real
            reg_synth = RandomForestRegressor(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )
            reg_synth.fit(X_synth, y_synth)
            tstr_r2 = r2_score(y_test, reg_synth.predict(X_test))
            
            # Handle negative R² (model worse than mean)
            if baseline_r2 <= 0:
                return 0.5
            
            # Ratio (capped at 1.0, handle negative TSTR R²)
            if tstr_r2 < 0:
                return 0.0
            return min(tstr_r2 / baseline_r2, 1.0)
            
        except Exception:
            return None
    
    def __repr__(self) -> str:
        return f"UtilityMetrics(random_state={self.random_state})"
