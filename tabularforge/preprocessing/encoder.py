"""
Data Encoder Module
===================

This module provides the DataEncoder class which handles automatic detection
and encoding of different column types in tabular data.

The encoder:
    1. Detects column types (categorical, numerical, datetime)
    2. Encodes categorical columns (label encoding, one-hot, etc.)
    3. Normalizes numerical columns
    4. Handles datetime columns
    5. Provides inverse transformations

Author: Sai Ganesh Kolan
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataEncoder:
    """
    Encoder for handling mixed-type tabular data.
    
    This class automatically detects column types and provides encoding
    and decoding functionality for preparing data for synthetic generation.
    
    Column Type Detection:
        - Categorical: Object/string dtype or low cardinality (<50 unique values)
        - Numerical: Int/float dtype with high cardinality
        - DateTime: Datetime dtype or parseable date strings
    
    Attributes:
        categorical_columns (List[str]): Names of categorical columns
        numerical_columns (List[str]): Names of numerical columns
        datetime_columns (List[str]): Names of datetime columns
        
    Example:
        >>> encoder = DataEncoder()
        >>> encoder.fit(data)
        >>> print(encoder.categorical_columns)
        ['gender', 'country']
        >>> print(encoder.numerical_columns)
        ['age', 'income']
    """
    
    # Threshold for detecting categorical columns
    # If a numerical column has fewer than this many unique values, treat as categorical
    CATEGORICAL_THRESHOLD = 50
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the DataEncoder.
        
        Args:
            categorical_columns: Explicit list of categorical column names.
                If None, will be auto-detected.
            numerical_columns: Explicit list of numerical column names.
                If None, will be auto-detected.
            datetime_columns: Explicit list of datetime column names.
                If None, will be auto-detected.
        """
        # Store user-specified column types
        self._user_categorical = categorical_columns
        self._user_numerical = numerical_columns
        self._user_datetime = datetime_columns
        
        # These will be set during fit()
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.datetime_columns: List[str] = []
        
        # Encoders for each column type
        # Dict mapping column name to encoder
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        
        # Store column order for reconstruction
        self._column_order: List[str] = []
        
        # Store statistics for numerical columns
        self._numerical_stats: Dict[str, Dict[str, float]] = {}
        
        # Track if encoder has been fitted
        self._is_fitted: bool = False
    
    def fit(self, data: pd.DataFrame) -> "DataEncoder":
        """
        Fit the encoder to the data.
        
        This method:
        1. Detects column types (if not specified)
        2. Fits label encoders for categorical columns
        3. Fits scalers for numerical columns
        4. Stores column statistics
        
        Args:
            data: DataFrame to fit the encoder on
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If data is empty
        """
        # =====================================================================
        # VALIDATION
        # =====================================================================
        
        if data.empty:
            raise ValueError("Cannot fit encoder on empty DataFrame")
        
        # Store column order
        self._column_order = list(data.columns)
        
        # =====================================================================
        # STEP 1: DETECT COLUMN TYPES
        # =====================================================================
        
        self._detect_column_types(data)
        
        # =====================================================================
        # STEP 2: FIT CATEGORICAL ENCODERS
        # =====================================================================
        
        for col in self.categorical_columns:
            # Create and fit label encoder
            encoder = LabelEncoder()
            
            # Handle missing values by converting to string
            col_data = data[col].fillna("__MISSING__").astype(str)
            encoder.fit(col_data)
            
            self._label_encoders[col] = encoder
        
        # =====================================================================
        # STEP 3: FIT NUMERICAL SCALERS
        # =====================================================================
        
        for col in self.numerical_columns:
            # Create and fit scaler
            scaler = StandardScaler()
            
            # Get non-null values
            col_data = data[col].dropna().values.reshape(-1, 1)
            
            if len(col_data) > 0:
                scaler.fit(col_data)
                
                # Store statistics
                self._numerical_stats[col] = {
                    "mean": float(scaler.mean_[0]),
                    "std": float(scaler.scale_[0]),
                    "min": float(data[col].min()),
                    "max": float(data[col].max())
                }
            else:
                # Handle edge case of all nulls
                self._numerical_stats[col] = {
                    "mean": 0.0,
                    "std": 1.0,
                    "min": 0.0,
                    "max": 1.0
                }
            
            self._scalers[col] = scaler
        
        self._is_fitted = True
        return self
    
    def _detect_column_types(self, data: pd.DataFrame) -> None:
        """
        Detect column types from the data.
        
        Detection logic:
        1. If user specified column types, use those
        2. Otherwise, auto-detect based on dtype and cardinality
        
        Args:
            data: DataFrame to analyze
        """
        # Use user-specified columns if provided
        if self._user_categorical is not None:
            self.categorical_columns = list(self._user_categorical)
        else:
            self.categorical_columns = []
        
        if self._user_numerical is not None:
            self.numerical_columns = list(self._user_numerical)
        else:
            self.numerical_columns = []
        
        if self._user_datetime is not None:
            self.datetime_columns = list(self._user_datetime)
        else:
            self.datetime_columns = []
        
        # If user specified ALL column types, don't auto-detect
        all_specified = (
            self._user_categorical is not None and 
            self._user_numerical is not None
        )
        
        if all_specified:
            return
        
        # Get columns that haven't been assigned yet
        assigned_columns = set(
            self.categorical_columns + 
            self.numerical_columns + 
            self.datetime_columns
        )
        unassigned = [col for col in data.columns if col not in assigned_columns]
        
        # Auto-detect types for unassigned columns
        for col in unassigned:
            dtype = data[col].dtype
            n_unique = data[col].nunique()
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(dtype):
                self.datetime_columns.append(col)
            
            # Check for categorical (object dtype or low cardinality)
            elif dtype == object or dtype.name == "category":
                self.categorical_columns.append(col)
            
            # Check for numerical with low cardinality (treat as categorical)
            elif n_unique <= self.CATEGORICAL_THRESHOLD:
                self.categorical_columns.append(col)
            
            # Otherwise, treat as numerical
            else:
                self.numerical_columns.append(col)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted encoders.
        
        This method:
        1. Encodes categorical columns to integers
        2. Normalizes numerical columns to zero mean, unit variance
        3. Converts datetime to numerical (timestamps)
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Transformed DataFrame (copy, original not modified)
            
        Raises:
            RuntimeError: If encoder has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder has not been fitted. Call fit() first.")
        
        # Make a copy to avoid modifying original
        transformed = data.copy()
        
        # Transform categorical columns
        for col in self.categorical_columns:
            if col in transformed.columns:
                col_data = transformed[col].fillna("__MISSING__").astype(str)
                
                # Handle unseen categories
                encoder = self._label_encoders[col]
                known_classes = set(encoder.classes_)
                col_data = col_data.map(
                    lambda x: x if x in known_classes else encoder.classes_[0]
                )
                
                transformed[col] = encoder.transform(col_data)
        
        # Transform numerical columns
        for col in self.numerical_columns:
            if col in transformed.columns:
                col_data = transformed[col].fillna(self._numerical_stats[col]["mean"])
                transformed[col] = self._scalers[col].transform(
                    col_data.values.reshape(-1, 1)
                ).flatten()
        
        # Transform datetime columns (convert to timestamp)
        for col in self.datetime_columns:
            if col in transformed.columns:
                transformed[col] = pd.to_datetime(transformed[col]).astype(np.int64)
        
        return transformed
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data back to original format.
        
        This method reverses the encoding:
        1. Decodes categorical integers back to original labels
        2. Denormalizes numerical columns
        3. Converts timestamps back to datetime
        
        Args:
            data: Transformed DataFrame
            
        Returns:
            DataFrame in original format
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder has not been fitted. Call fit() first.")
        
        # Make a copy
        original = data.copy()
        
        # Inverse transform categorical columns
        for col in self.categorical_columns:
            if col in original.columns:
                encoder = self._label_encoders[col]
                n_classes = len(encoder.classes_)
                
                # Clip to valid range and round
                col_data = np.clip(
                    np.round(original[col].values), 
                    0, 
                    n_classes - 1
                ).astype(int)
                
                original[col] = encoder.inverse_transform(col_data)
                
                # Replace __MISSING__ back to NaN
                original[col] = original[col].replace("__MISSING__", np.nan)
        
        # Inverse transform numerical columns
        for col in self.numerical_columns:
            if col in original.columns:
                original[col] = self._scalers[col].inverse_transform(
                    original[col].values.reshape(-1, 1)
                ).flatten()
        
        # Inverse transform datetime columns
        for col in self.datetime_columns:
            if col in original.columns:
                original[col] = pd.to_datetime(original[col].astype(np.int64))
        
        return original
    
    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all columns.
        
        Returns:
            Dictionary with column names as keys and info dicts as values
        """
        info = {}
        
        for col in self.categorical_columns:
            info[col] = {
                "type": "categorical",
                "n_categories": len(self._label_encoders[col].classes_),
                "categories": list(self._label_encoders[col].classes_)
            }
        
        for col in self.numerical_columns:
            info[col] = {
                "type": "numerical",
                **self._numerical_stats.get(col, {})
            }
        
        for col in self.datetime_columns:
            info[col] = {
                "type": "datetime"
            }
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (
            f"DataEncoder("
            f"categorical={len(self.categorical_columns)}, "
            f"numerical={len(self.numerical_columns)}, "
            f"datetime={len(self.datetime_columns)})"
        )