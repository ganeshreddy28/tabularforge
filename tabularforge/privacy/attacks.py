"""
Privacy Attack Simulation Module

This module provides tools for simulating privacy attacks against
synthetic data. This helps evaluate how well the synthetic data
protects the privacy of individuals in the original dataset.

Supported Attacks:
    1. Membership Inference: Can an attacker determine if a specific
       record was in the training data?
    2. Attribute Inference: Can an attacker infer sensitive attributes
       given other information about a record?
    3. Nearest Neighbor: How close are synthetic records to real ones?

Author: Sai Ganesh Kolan
License: MIT
"""

# IMPORTS
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class PrivacyAttackSimulator:
    """
    Simulates privacy attacks against synthetic data.
    
    This class helps evaluate the privacy protection of synthetic data
    by simulating common privacy attacks. Lower attack success rates
    indicate better privacy protection.
    
    Attacks:
        - Membership Inference: Tests if attacker can tell if a record
          was in the training data. Ideal success rate: 50% (random guess)
        - Attribute Inference: Tests if attacker can infer sensitive
          attributes. Lower is better.
        - Nearest Neighbor Distance: Measures how close synthetic records
          are to real ones. Higher distance = better privacy.
    
    Example:
        >>> simulator = PrivacyAttackSimulator()
        >>> results = simulator.run_all_attacks(real_data, synthetic_data)
        >>> print(f"Membership inference risk: {results['membership_inference']:.2%}")
    """
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize the attack simulator.
        
        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def membership_inference_attack(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        test_size: float = 0.3
    ) -> float:
        """
        Simulate a membership inference attack.
        
        This attack tests whether an attacker can determine if a specific
        record was used to train the synthetic data generator.
        
        Method:
            1. Split real data into "train" (used for generation) and "test" (holdout)
            2. Train a classifier to distinguish records in "train" from synthetic
            3. The success rate indicates privacy risk
        
        Interpretation:
            - 0.5 (50%): Perfect privacy (attacker is guessing randomly)
            - 1.0 (100%): No privacy (attacker can perfectly identify)
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            test_size: Fraction of real data to use as holdout
            
        Returns:
            Attack success rate (0.0 to 1.0). Lower is better.
        """
        # Ensure same columns
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        real = real_data[common_cols].copy()
        synth = synthetic_data[common_cols].copy()
        
        # Convert to numeric
        for col in common_cols:
            if real[col].dtype == object:
                real[col] = pd.Categorical(real[col]).codes
                synth[col] = pd.Categorical(synth[col]).codes
        
        # Fill NaN
        real = real.fillna(0)
        synth = synth.fillna(0)
        
        # Split real data
        real_train, real_test = train_test_split(
            real, test_size=test_size, random_state=self.random_state
        )
        
        # Create dataset: real_train (label=1) vs synthetic (label=0)
        # Use same number of synthetic samples as real_train samples
        n_samples = min(len(real_train), len(synth))
        X_train = pd.concat([real_train.iloc[:n_samples], synth.iloc[:n_samples]])
        y_train = np.array([1] * n_samples + [0] * n_samples)
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # Test on holdout real data
        # If classifier thinks holdout is "from training" at high rate, 
        # it means the model memorized
        predictions = clf.predict(real_test)
        
        # Success rate = fraction classified as "from training" (label=1)
        # High rate means attacker can identify training members
        success_rate = np.mean(predictions)
        
        return success_rate
    
    def attribute_inference_attack(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_column: str
    ) -> float:
        """
        Simulate an attribute inference attack.
        
        This attack tests whether an attacker can infer the value of a
        sensitive attribute given other attributes.
        
        Method:
            1. Train a model on synthetic data to predict sensitive attribute
            2. Test on real data
            3. High accuracy indicates the synthetic data leaks information
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            sensitive_column: Name of the sensitive column to infer
            
        Returns:
            Attack success rate (accuracy). Lower is better.
        """
        if sensitive_column not in real_data.columns:
            raise ValueError(f"Column '{sensitive_column}' not found")
        
        # Features = all columns except sensitive
        feature_cols = [c for c in real_data.columns if c != sensitive_column]
        
        # Prepare data
        X_synth = synthetic_data[feature_cols].copy()
        y_synth = synthetic_data[sensitive_column].copy()
        X_real = real_data[feature_cols].copy()
        y_real = real_data[sensitive_column].copy()
        
        # Convert to numeric
        for col in feature_cols:
            if X_synth[col].dtype == object:
                X_synth[col] = pd.Categorical(X_synth[col]).codes
                X_real[col] = pd.Categorical(X_real[col]).codes
        
        if y_synth.dtype == object:
            y_synth = pd.Categorical(y_synth).codes
            y_real = pd.Categorical(y_real).codes
        
        # Fill NaN
        X_synth = X_synth.fillna(0)
        X_real = X_real.fillna(0)
        
        # Train on synthetic, test on real
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_synth, y_synth)
        
        # Accuracy on real data
        accuracy = clf.score(X_real, y_real)
        
        return accuracy
    
    def nearest_neighbor_distance(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        n_neighbors: int = 1
    ) -> Dict[str, float]:
        """
        Calculate distance to nearest real record.
        
        This measures how close each synthetic record is to its nearest
        real record. Very close records might indicate memorization.
        
        Interpretation:
            - Higher average distance = better privacy
            - If some synthetic records are extremely close to real ones,
              they might be "copies" of real records
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            n_neighbors: Number of nearest neighbors to consider
            
        Returns:
            Dictionary with distance statistics
        """
        # Ensure same columns
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        real = real_data[common_cols].copy()
        synth = synthetic_data[common_cols].copy()
        
        # Convert to numeric
        for col in common_cols:
            if real[col].dtype == object:
                real[col] = pd.Categorical(real[col]).codes
                synth[col] = pd.Categorical(synth[col]).codes
        
        # Fill NaN and convert to arrays
        real_arr = real.fillna(0).values.astype(float)
        synth_arr = synth.fillna(0).values.astype(float)
        
        # Normalize for fair distance comparison
        mean = real_arr.mean(axis=0)
        std = real_arr.std(axis=0) + 1e-6
        real_arr = (real_arr - mean) / std
        synth_arr = (synth_arr - mean) / std
        
        # Fit nearest neighbors on real data
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(real_arr)
        
        # Find distances from synthetic to nearest real
        distances, _ = nn.kneighbors(synth_arr)
        
        return {
            "mean_distance": float(np.mean(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "median_distance": float(np.median(distances)),
            "std_distance": float(np.std(distances))
        }
    
    def run_all_attacks(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run all privacy attacks and return comprehensive results.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            sensitive_column: Optional column for attribute inference
            
        Returns:
            Dictionary with all attack results
        """
        results = {}
        
        # Membership inference
        results["membership_inference_risk"] = self.membership_inference_attack(
            real_data, synthetic_data
        )
        
        # Attribute inference (if sensitive column specified)
        if sensitive_column and sensitive_column in real_data.columns:
            results["attribute_inference_risk"] = self.attribute_inference_attack(
                real_data, synthetic_data, sensitive_column
            )
        
        # Nearest neighbor distance
        nn_results = self.nearest_neighbor_distance(real_data, synthetic_data)
        results["nearest_neighbor_distance"] = nn_results["mean_distance"]
        results["min_neighbor_distance"] = nn_results["min_distance"]
        
        return results
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PrivacyAttackSimulator(random_state={self.random_state})"