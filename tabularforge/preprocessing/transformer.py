"""
Data Transformer Module

This module provides the DataTransformer class which orchestrates
the full data transformation pipeline for synthetic data generation.

The transformer coordinates:
    1. Encoding (using DataEncoder)
    2. Additional transformations (e.g., mode-specific normalization)
    3. Inverse transformations for generated data

Author: Sai Ganesh Kolan
License: MIT
"""

# IMPORTS
from typing import Any, Optional

import numpy as np
import pandas as pd

from tabularforge.preprocessing.encoder import DataEncoder


class DataTransformer:
    """
    Orchestrates the complete data transformation pipeline.
    
    The DataTransformer is responsible for:
    1. Coordinating with the DataEncoder for type-based encoding
    2. Applying any additional transformations needed by generators
    3. Reversing all transformations for the final output
    
    This class acts as the single point of contact for data transformation,
    making it easy to add or modify transformation steps without changing
    the generator code.
    
    Attributes:
        _is_fitted (bool): Whether the transformer has been fitted
        _original_columns (List[str]): Original column order
        _original_dtypes (Dict): Original column data types
        
    Example:
        >>> transformer = DataTransformer()
        >>> transformed = transformer.fit_transform(data, encoder)
        >>> # ... generate synthetic data ...
        >>> original_format = transformer.inverse_transform(synthetic, encoder)
    """
    
    def __init__(self) -> None:
        """
        Initialize the DataTransformer.
        
        The transformer is initialized without any fitted state.
        Call fit() or fit_transform() to prepare it for use.
        """
        # Track fitted state
        self._is_fitted: bool = False
        
        # Store original column information for reconstruction
        self._original_columns: list = []
        self._original_dtypes: dict = {}
    
    def fit(self, data: pd.DataFrame, encoder: DataEncoder) -> "DataTransformer":
        """
        Fit the transformer to the data.
        
        This method stores information about the original data format
        that will be needed when inverse transforming generated data.
        
        Args:
            data: Original DataFrame to fit on
            encoder: Fitted DataEncoder with column type information
            
        Returns:
            self (for method chaining)
        """
        # Store original column order and dtypes
        self._original_columns = list(data.columns)
        self._original_dtypes = {col: data[col].dtype for col in data.columns}
        
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, encoder: DataEncoder) -> pd.DataFrame:
        """
        Transform data for use with synthetic data generators.
        
        This method applies all necessary transformations to prepare
        the data for the generator:
        1. Type-based encoding (via encoder)
        2. Any additional transformations
        
        Args:
            data: DataFrame to transform
            encoder: Fitted DataEncoder
            
        Returns:
            Transformed DataFrame ready for generator
        """
        # Apply encoder transformation
        transformed = encoder.transform(data)
        
        # Ensure all columns are numeric (required for most generators)
        for col in transformed.columns:
            if transformed[col].dtype == object:
                # This shouldn't happen if encoder is working correctly
                # but handle it gracefully
                transformed[col] = pd.to_numeric(transformed[col], errors='coerce')
        
        # Fill any remaining NaN values
        transformed = transformed.fillna(0)
        
        return transformed
    
    def fit_transform(self, data: pd.DataFrame, encoder: DataEncoder) -> pd.DataFrame:
        """
        Fit the transformer and transform the data in one step.
        
        This is a convenience method that combines fit() and transform().
        
        Args:
            data: DataFrame to fit on and transform
            encoder: Fitted DataEncoder
            
        Returns:
            Transformed DataFrame
        """
        self.fit(data, encoder)
        return self.transform(data, encoder)
    
    def inverse_transform(
        self, 
        data: pd.DataFrame, 
        encoder: DataEncoder
    ) -> pd.DataFrame:
        """
        Inverse transform generated data back to original format.
        
        This method reverses all transformations to produce data in
        the same format as the original input data.
        
        Args:
            data: Transformed DataFrame (e.g., from generator)
            encoder: The same encoder used during transform
            
        Returns:
            DataFrame in original format
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Transformer has not been fitted. Call fit() or fit_transform() first."
            )
        
        # Apply encoder inverse transform
        original_format = encoder.inverse_transform(data)
        
        # Ensure column order matches original
        # Only keep columns that were in the original data
        final_columns = [col for col in self._original_columns if col in original_format.columns]
        original_format = original_format[final_columns]
        
        # Try to restore original dtypes where possible
        for col in original_format.columns:
            if col in self._original_dtypes:
                try:
                    original_dtype = self._original_dtypes[col]
                    
                    # Handle integer columns
                    if pd.api.types.is_integer_dtype(original_dtype):
                        original_format[col] = original_format[col].round().astype(original_dtype)
                    
                    # Handle float columns (already float, usually fine)
                    elif pd.api.types.is_float_dtype(original_dtype):
                        original_format[col] = original_format[col].astype(original_dtype)
                    
                except (ValueError, TypeError):
                    # If conversion fails, keep as is
                    pass
        
        return original_format
    
    def __repr__(self) -> str:
        """String representation of the transformer."""
        status = "fitted" if self._is_fitted else "not fitted"
        n_cols = len(self._original_columns) if self._is_fitted else 0
        return f"DataTransformer({status}, n_columns={n_cols})"
