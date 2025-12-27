"""
Base Generator Module
---------------------

This module defines the abstract base class for all synthetic data generators.
All generator implementations (Copula, CTGAN, TVAE, etc.) must inherit from
this class and implement its abstract methods.

Design Pattern: Template Method
    The BaseGenerator uses the Template Method pattern. It defines the skeleton
    of the generation algorithm, with specific steps delegated to subclasses.
    This ensures all generators have a consistent interface.

Author: Sai Ganesh Kolan
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class BaseGenerator(ABC):
    """
    Abstract base class for all synthetic data generators.
    
    This class defines the interface that all generators must implement.
    It provides a consistent API for fitting generators to data and
    generating new synthetic samples.
    
    All subclasses must implement:
        - _fit(): Internal fitting logic
        - _sample(): Internal sampling logic
        
    The public methods fit() and sample() handle common preprocessing
    and postprocessing, then delegate to the internal methods.
    
    Attributes:
        random_state (int): Seed for reproducibility
        is_fitted (bool): Whether the generator has been fitted
        _column_info (Dict): Information about columns (types, stats)
        
    Example:
        >>> # This is an abstract class - use a concrete implementation:
        >>> from tabularforge.generators import GaussianCopulaGenerator
        >>> generator = GaussianCopulaGenerator()
        >>> generator.fit(data, encoder)
        >>> synthetic = generator.sample(1000)
    """
    
    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize the base generator.
        
        Args:
            random_state (int, optional): 
                Seed for random number generator. Use for reproducible results.
                If None, results will vary between runs.
        """
        # Store the random state for reproducibility
        # This will be used to seed numpy and torch random generators
        self.random_state: Optional[int] = random_state
        
        # Flag to track whether the generator has been fitted
        # Prevents calling sample() before fit()
        self.is_fitted: bool = False
        
        # Dictionary to store column information learned during fitting
        # Keys are column names, values contain type, statistics, etc.
        self._column_info: Dict[str, Any] = {}
        
        # Store the encoder reference for use during sampling
        self._encoder: Optional[Any] = None
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        encoder: Any
    ) -> "BaseGenerator":
        """
        Fit the generator to the training data.
        
        This method learns the statistical properties of the data that are
        needed to generate new synthetic samples. The specific properties
        learned depend on the generator type.
        
        Args:
            data (pd.DataFrame or np.ndarray): 
                Training data to learn from. Should be preprocessed/transformed.
                
            encoder (DataEncoder): 
                The encoder that was used to transform the data.
                Contains information about column types.
                
        Returns:
            BaseGenerator: 
                Returns self to allow method chaining:
                generator.fit(data, encoder).sample(100)
                
        Raises:
            ValueError: If data is empty or has invalid format.
        """
        # =====================================================================
        # VALIDATION
        # =====================================================================
        
        # Check that data is not empty
        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError("Cannot fit on empty DataFrame")
        if isinstance(data, np.ndarray) and data.size == 0:
            raise ValueError("Cannot fit on empty array")
        
        # =====================================================================
        # STORE ENCODER AND COLUMN INFO
        # =====================================================================
        
        # Store encoder for use during sampling
        self._encoder = encoder
        
        # Store column information
        if isinstance(data, pd.DataFrame):
            self._column_info = {
                col: {
                    "dtype": str(data[col].dtype),
                    "n_unique": data[col].nunique()
                }
                for col in data.columns
            }
        
        # =====================================================================
        # DELEGATE TO SUBCLASS IMPLEMENTATION
        # =====================================================================
        
        # Call the subclass-specific fitting logic
        self._fit(data, encoder)
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    @abstractmethod
    def _fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        encoder: Any
    ) -> None:
        """
        Internal fitting logic - must be implemented by subclasses.
        
        This method should learn whatever statistical properties of the data
        are needed by the specific generator type.
        
        Args:
            data: Training data (preprocessed)
            encoder: Column encoder with type information
        """
        pass  # Subclasses must implement this
    
    def sample(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.
        
        This method generates new synthetic data points that statistically
        resemble the training data.
        
        Args:
            n_samples (int): 
                Number of samples to generate. Must be positive.
                
            conditions (Dict[str, Any], optional): 
                Conditions for conditional generation. The format depends
                on the specific generator implementation.
                
        Returns:
            pd.DataFrame: 
                DataFrame containing n_samples rows of synthetic data.
                
        Raises:
            RuntimeError: If generator has not been fitted.
            ValueError: If n_samples is not positive.
        """
        # =====================================================================
        # VALIDATION
        # =====================================================================
        
        # Check that generator has been fitted
        if not self.is_fitted:
            raise RuntimeError(
                "Generator has not been fitted. Call fit() before sample()."
            )
        
        # Validate n_samples
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        # =====================================================================
        # DELEGATE TO SUBCLASS IMPLEMENTATION
        # =====================================================================
        
        # Call the subclass-specific sampling logic
        synthetic_data = self._sample(n_samples, conditions)
        
        return synthetic_data
    
    @abstractmethod
    def _sample(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Internal sampling logic - must be implemented by subclasses.
        
        This method should generate new samples based on the learned
        statistical properties.
        
        Args:
            n_samples: Number of samples to generate
            conditions: Optional conditions for conditional generation
            
        Returns:
            DataFrame with synthetic samples
        """
        pass  # Subclasses must implement this
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get generator parameters.
        
        Returns a dictionary of all parameters used by this generator,
        useful for reproducibility and logging.
        
        Returns:
            Dict[str, Any]: Dictionary of parameter names and values.
        """
        return {
            "random_state": self.random_state,
            "is_fitted": self.is_fitted
        }
    
    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return f"{self.__class__.__name__}(random_state={self.random_state})"
