"""
Gaussian Copula Generator

This module implements synthetic data generation using Gaussian Copulas.
Copulas are mathematical functions that describe the dependence structure
between random variables, separate from their marginal distributions.

How It Works:
    1. Learn the marginal distribution of each column
    2. Transform each column to uniform using its CDF (Cumulative Distribution Function)
    3. Transform uniforms to standard normal using the inverse normal CDF
    4. Fit a multivariate Gaussian to capture correlations
    5. For sampling: reverse the process

Advantages:
    - Fast training and sampling
    - Good at preserving column correlations
    - Works well for data with Gaussian-like distributions
    
Disadvantages:
    - May not capture complex, non-linear relationships
    - Assumes continuous underlying distributions

Author: Sai Ganesh Kolan
License: MIT
"""

# IMPORTS
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, rankdata

from tabularforge.generators.base import BaseGenerator


class GaussianCopulaGenerator(BaseGenerator):
    """
    Gaussian Copula-based synthetic data generator.
    
    This generator uses Gaussian Copulas to model the joint distribution of
    tabular data. It's particularly good at:
    - Preserving correlations between columns
    - Fast training and generation
    - Handling mixed data types (with proper encoding)
    
    The algorithm works by:
    1. Converting each column to its marginal CDF (uniform distribution)
    2. Converting uniforms to standard normal (Gaussian)
    3. Fitting a multivariate Gaussian to capture correlations
    4. Reversing the process to generate new samples
    
    Attributes:
        correlation_matrix (np.ndarray): Learned correlation matrix
        marginals (Dict): Learned marginal distributions for each column
        column_names (List[str]): Names of columns in training data
        
    Example:
        >>> generator = GaussianCopulaGenerator(random_state=42)
        >>> generator.fit(data, encoder)
        >>> synthetic = generator.sample(1000)
    """
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize the Gaussian Copula generator.
        
        Args:
            random_state (int, optional): 
                Seed for random number generator for reproducibility.
        """
        # Call parent class constructor
        super().__init__(random_state=random_state)
        
        # Initialise attributes that will be set during fitting
        
        # Correlation matrix of the Gaussian copula
        # Shape: (n_columns, n_columns)
        self.correlation_matrix: Optional[np.ndarray] = None
        
        # Dictionary storing marginal distribution info for each column
        # Key: column name, Value: dict with distribution parameters
        self.marginals: Dict[str, Dict[str, Any]] = {}
        
        # List of column names (to preserve order)
        self.column_names: List[str] = []
        
        # Number of columns
        self._n_columns: int = 0
        
        # Mean vector for the multivariate Gaussian (should be zeros)
        self._mean: Optional[np.ndarray] = None
    
    def _fit(
        self,
        data: pd.DataFrame,
        encoder: Any
    ) -> None:
        """
        Fit the Gaussian Copula to the training data.
        
        This method learns:
        1. The marginal distribution of each column
        2. The correlation structure between columns
        
        Args:
            data (pd.DataFrame): 
                Training data (already transformed by encoder)
            encoder: 
                Data encoder with column type information
        """
        # STEP 1: STORE BASIC INFO
        
        # Store column names and count
        self.column_names = list(data.columns)
        self._n_columns = len(self.column_names)
        
        # STEP 2: LEARN MARGINAL DISTRIBUTIONS
        # For each column, we learn its empirical distribution
        # This allows us to transform the column to uniform later
        
        for col in self.column_names:
            # Get column values
            col_data = data[col].values
            
            # Store marginal info
            # We use the empirical distribution (sorted values for ECDF)
            self.marginals[col] = {
                # Sorted unique values for empirical CDF
                "values": np.sort(col_data),
                # Mean and std for parametric approximation
                "mean": np.mean(col_data),
                "std": np.std(col_data) + 1e-6,  # Add small epsilon to avoid division by zero
                # Min and max for bounds
                "min": np.min(col_data),
                "max": np.max(col_data),
                # Number of samples
                "n_samples": len(col_data)
            }
        
        # STEP 3: TRANSFORM TO UNIFORM SPACE (USING ECDF)
        # Transform each column to uniform [0, 1] using empirical CDF
        
        uniform_data = np.zeros_like(data.values, dtype=float)
        
        for i, col in enumerate(self.column_names):
            col_data = data[col].values
            
            # Use rank-based transformation to get uniform distribution
            # rankdata gives ranks 1 to n, divide by (n+1) to get (0, 1)
            ranks = rankdata(col_data, method='average')
            uniform_data[:, i] = ranks / (len(col_data) + 1)
        
        # STEP 4: TRANSFORM TO NORMAL SPACE
        # Apply inverse normal CDF (percent point function) to get standard normal
        
        # Clip to avoid infinity at 0 and 1
        uniform_data = np.clip(uniform_data, 1e-6, 1 - 1e-6)
        
        # Transform uniform to normal
        normal_data = norm.ppf(uniform_data)
        
        
        # STEP 5: ESTIMATE CORRELATION MATRIX
        # Fit multivariate Gaussian by estimating the correlation matrix
        
        # Calculate correlation matrix
        # We use np.corrcoef which returns correlation (not covariance)
        self.correlation_matrix = np.corrcoef(normal_data, rowvar=False)
        
        # Ensure correlation matrix is positive semi-definite
        # (numerical issues can sometimes make it slightly non-PSD)
        self.correlation_matrix = self._nearest_positive_definite(
            self.correlation_matrix
        )
        
        # Mean vector should be zeros for standard normal
        self._mean = np.zeros(self._n_columns)
        
    def _sample(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic samples using the fitted Gaussian Copula.
        
        The sampling process reverses the fitting process:
        1. Sample from multivariate Gaussian with learned correlation
        2. Transform Gaussian samples to uniform using normal CDF
        3. Transform uniform to original space using inverse marginal CDF
        
        Args:
            n_samples (int): 
                Number of samples to generate
            conditions (Dict, optional): 
                Not fully supported yet for copula
                
        Returns:
            pd.DataFrame: 
                DataFrame with n_samples rows of synthetic data
        """
        # STEP 1: SAMPLE FROM MULTIVARIATE GAUSSIAN
        # Generate samples from multivariate normal with learned correlations
        
        # Set random state if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Sample from multivariate normal
        # This gives us correlated standard normal values
        normal_samples = np.random.multivariate_normal(
            mean=self._mean,
            cov=self.correlation_matrix,
            size=n_samples
        )
        
        # STEP 2: TRANSFORM TO UNIFORM SPACE
        # Apply normal CDF to get uniform [0, 1] values
        
        uniform_samples = norm.cdf(normal_samples)
        
        # STEP 3: TRANSFORM TO ORIGINAL SPACE
        # Use inverse empirical CDF to get values in original distribution
        
        synthetic_data = np.zeros_like(uniform_samples)
        
        for i, col in enumerate(self.column_names):
            marginal = self.marginals[col]
            
            # Use quantile function (inverse CDF)
            # For empirical distribution: interpolate sorted values
            n_stored = marginal["n_samples"]
            indices = uniform_samples[:, i] * (n_stored - 1)
            lower_idx = np.floor(indices).astype(int)
            upper_idx = np.ceil(indices).astype(int)
            
            # Clip to valid range
            lower_idx = np.clip(lower_idx, 0, n_stored - 1)
            upper_idx = np.clip(upper_idx, 0, n_stored - 1)
            
            # Linear interpolation between nearest stored values
            frac = indices - lower_idx
            lower_vals = marginal["values"][lower_idx]
            upper_vals = marginal["values"][upper_idx]
            synthetic_data[:, i] = lower_vals + frac * (upper_vals - lower_vals)
        
        # STEP 4: CREATE DATAFRAME
        
        synthetic_df = pd.DataFrame(
            synthetic_data,
            columns=self.column_names
        )
        
        return synthetic_df
    
    def _nearest_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """
        Find the nearest positive-definite matrix to the input.
        
        Correlation matrices should be positive semi-definite, but numerical
        errors can sometimes result in matrices that are slightly non-PSD.
        This method fixes such matrices.
        
        Algorithm:
            Uses eigenvalue decomposition. Negative eigenvalues are replaced
            with small positive values, then the matrix is reconstructed.
        
        Args:
            matrix (np.ndarray): 
                Input matrix (should be symmetric)
                
        Returns:
            np.ndarray: 
                Nearest positive-definite matrix
        """
        # Compute eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Replace negative eigenvalues with small positive value
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        
        # Reconstruct matrix
        psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Ensure symmetry (correct for numerical errors)
        psd_matrix = (psd_matrix + psd_matrix.T) / 2
        
        # Ensure diagonal is exactly 1 (for correlation matrix)
        d = np.sqrt(np.diag(psd_matrix))
        psd_matrix = psd_matrix / np.outer(d, d)
        
        return psd_matrix
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get generator parameters.
        
        Returns:
            Dict containing all parameters and learned values.
        """
        params = super().get_params()
        params.update({
            "n_columns": self._n_columns,
            "column_names": self.column_names,
            "has_correlation_matrix": self.correlation_matrix is not None
        })
        return params
