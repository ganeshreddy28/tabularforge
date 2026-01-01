"""
TabularForge Main Class

This module contains the TabularForge class, which is the primary interface
for generating synthetic tabular data. It provides a simple, unified API
that abstracts away the complexity of different generators and preprocessing.

Example Usage:
    >>> from tabularforge import TabularForge
    >>> import pandas as pd
    >>> 
    >>> real_data = pd.read_csv("data.csv")
    >>> forge = TabularForge(real_data)
    >>> synthetic = forge.generate(n_samples=1000)

Author: Sai Ganesh Kolan
License: MIT
"""

# IMPORTS
# Standard library imports (built into Python)
import logging
from typing import Dict, List, Optional, Union, Any

# Third-party imports (installed via pip)
import numpy as np
import pandas as pd

# Local imports (from this package)
from tabularforge.generators.base import BaseGenerator
from tabularforge.generators.copula import GaussianCopulaGenerator
from tabularforge.generators.ctgan import CTGANGenerator
from tabularforge.generators.tvae import TVAEGenerator
from tabularforge.preprocessing.encoder import DataEncoder
from tabularforge.preprocessing.transformer import DataTransformer
from tabularforge.privacy.differential import DifferentialPrivacy
from tabularforge.metrics.statistical import StatisticalMetrics
from tabularforge.metrics.utility import UtilityMetrics
from tabularforge.metrics.privacy import PrivacyMetrics

# LOGGER SETUP
# Create a logger specific to this module
# This allows users to control logging verbosity for this module specifically
logger = logging.getLogger(__name__)


# GENERATOR REGISTRY
# This dictionary maps generator names (strings) to their class implementations.
# This allows users to specify generators by name: TabularForge(data, generator='ctgan')
# Adding a new generator is as simple as adding an entry here.
AVAILABLE_GENERATORS: Dict[str, type] = {
    "copula": GaussianCopulaGenerator,      # Fast, good for simple distributions
    "gaussian_copula": GaussianCopulaGenerator,  # Alias for copula
    "ctgan": CTGANGenerator,                 # Good for complex relationships
    "tvae": TVAEGenerator,                   # Good for high-dimensional data
}


class TabularForge:
    """
    TabularForge: Privacy-Preserving Synthetic Tabular Data Generation
    
    This is the main class that users interact with. It provides a simple,
    unified interface for:
    - Fitting a synthetic data generator to real data
    - Generating new synthetic samples
    - Evaluating quality and privacy of synthetic data
    - Benchmarking different generators
    
    Attributes:
        data (pd.DataFrame): The original real data
        generator (BaseGenerator): The fitted synthetic data generator
        encoder (DataEncoder): Handles encoding/decoding of columns
        transformer (DataTransformer): Handles data transformations
        privacy (DifferentialPrivacy): Optional privacy mechanism
        
    Example:
        >>> # Basic usage
        >>> forge = TabularForge(real_data)
        >>> synthetic = forge.generate(n_samples=1000)
        
        >>> # With privacy
        >>> forge = TabularForge(real_data, privacy_epsilon=1.0)
        >>> private_synthetic = forge.generate(n_samples=1000)
        
        >>> # With specific generator
        >>> forge = TabularForge(real_data, generator='ctgan')
        >>> synthetic = forge.generate(n_samples=1000)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        generator: str = "copula",
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        privacy_epsilon: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: bool = True
    ) -> None:
        """
        Initialize TabularForge with real data.
        
        This constructor performs the following steps:
        1. Validates and stores the input data
        2. Detects column types (or uses provided ones)
        3. Initializes the encoder and transformer
        4. Initializes the selected generator
        5. Optionally sets up differential privacy
        6. Fits the generator to the data
        
        Args:
            data (pd.DataFrame): 
                The real tabular data to learn from. Must be a pandas DataFrame
                with at least one row and one column.
                
            generator (str, optional): 
                Name of the generator to use. Options are:
                - 'copula' or 'gaussian_copula': Fast, good for simple data
                - 'ctgan': Good for complex relationships
                - 'tvae': Good for high-dimensional data
                Defaults to 'copula'.
                
            categorical_columns (List[str], optional): 
                List of column names that should be treated as categorical.
                If None, will be auto-detected based on data types and cardinality.
                
            numerical_columns (List[str], optional): 
                List of column names that should be treated as numerical.
                If None, will be auto-detected.
                
            datetime_columns (List[str], optional): 
                List of column names that contain datetime values.
                If None, will be auto-detected.
                
            privacy_epsilon (float, optional): 
                Privacy budget for differential privacy. Lower values provide
                stronger privacy but may reduce data utility.
                - None: No differential privacy (default)
                - 0.1: Very strong privacy
                - 1.0: Balanced privacy/utility
                - 10.0: Weak privacy, high utility
                
            random_state (int, optional): 
                Seed for random number generator. Use for reproducible results.
                If None, results will vary between runs.
                
            verbose (bool, optional): 
                Whether to print progress messages. Defaults to True.
                
        Raises:
            ValueError: If data is empty or generator name is invalid.
            TypeError: If data is not a pandas DataFrame.
            
        Example:
            >>> # Auto-detect column types
            >>> forge = TabularForge(data)
            
            >>> # Specify column types explicitly
            >>> forge = TabularForge(
            ...     data,
            ...     categorical_columns=['gender', 'country'],
            ...     numerical_columns=['age', 'income']
            ... )
            
            >>> # With privacy and reproducibility
            >>> forge = TabularForge(
            ...     data,
            ...     privacy_epsilon=1.0,
            ...     random_state=42
            ... )
        """
        # STEP 1: INPUT VALIDATION
        # We validate inputs early to fail fast with clear error messages
        
        # Check that data is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(data).__name__}. "
                f"Convert your data to DataFrame first: pd.DataFrame(your_data)"
            )
        
        # Check that data is not empty
        if data.empty:
            raise ValueError(
                "Input data is empty. Please provide a DataFrame with at least "
                "one row and one column."
            )
        
        # Check that generator name is valid
        if generator.lower() not in AVAILABLE_GENERATORS:
            available = ", ".join(AVAILABLE_GENERATORS.keys())
            raise ValueError(
                f"Unknown generator '{generator}'. "
                f"Available generators: {available}"
            )
        
        # STEP 2: STORE CONFIGURATION
        # We store all configuration for later use and reproducibility
        
        # Store the original data (make a copy to avoid modifying user's data)
        self.data: pd.DataFrame = data.copy()
        
        # Store configuration parameters
        self._generator_name: str = generator.lower()
        self._privacy_epsilon: Optional[float] = privacy_epsilon
        self._random_state: Optional[int] = random_state
        self._verbose: bool = verbose
        
        # Set random seed if provided (for reproducibility)
        if random_state is not None:
            np.random.seed(random_state)
            
        # Log initialization
        if verbose:
            logger.info(f"Initializing TabularForge with {len(data)} rows, {len(data.columns)} columns")
            logger.info(f"Generator: {generator}")
        
        # STEP 3: COLUMN TYPE DETECTION
        # Detect or validate column types for proper preprocessing
        
        # Initialize the encoder which handles column type detection
        self.encoder: DataEncoder = DataEncoder(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            datetime_columns=datetime_columns
        )
        
        # Fit the encoder to learn column types and encoding schemes
        self.encoder.fit(self.data)
        
        if verbose:
            logger.info(f"Detected {len(self.encoder.categorical_columns)} categorical columns")
            logger.info(f"Detected {len(self.encoder.numerical_columns)} numerical columns")
        
        # STEP 4: DATA TRANSFORMATION
        # Transform data into format suitable for the generator
        
        # Initialize the transformer
        self.transformer: DataTransformer = DataTransformer()
        
        # Transform the data (encode categories, normalize numericals)
        transformed_data = self.transformer.fit_transform(
            self.data,
            self.encoder
        )
        
        # STEP 5: PRIVACY SETUP (OPTIONAL)
        # Set up differential privacy if epsilon is provided
        
        self.privacy: Optional[DifferentialPrivacy] = None
        if privacy_epsilon is not None:
            if verbose:
                logger.info(f"Setting up differential privacy with epsilon={privacy_epsilon}")
            self.privacy = DifferentialPrivacy(epsilon=privacy_epsilon)
        
        # STEP 6: GENERATOR INITIALIZATION AND FITTING
        # Create and fit the selected generator
        
        # Get the generator class from the registry
        generator_class = AVAILABLE_GENERATORS[self._generator_name]
        
        # Initialize the generator
        self.generator: BaseGenerator = generator_class(
            random_state=random_state
        )
        
        # Fit the generator to the transformed data
        if verbose:
            logger.info(f"Fitting {self._generator_name} generator...")
            
        self.generator.fit(transformed_data, self.encoder)
        
        if verbose:
            logger.info("TabularForge initialized successfully!")
        
        # STEP 7: STORE METADATA
        # Store information about the fitted model for later use
        
        self._is_fitted: bool = True
        self._n_original_samples: int = len(data)
        self._column_names: List[str] = list(data.columns)
    
    def generate(
        self,
        n_samples: int = 1000,
        conditions: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data samples.
        
        This method generates new synthetic samples that statistically resemble
        the original data while (optionally) preserving privacy.
        
        Args:
            n_samples (int, optional): 
                Number of synthetic samples to generate. Defaults to 1000.
                Must be a positive integer.
                
            conditions (Dict[str, Any], optional): 
                Conditions for conditional generation. Keys are column names,
                values are the conditions. Examples:
                - {'gender': 'female'}: Only generate female samples
                - {'age': '>30'}: Only generate samples with age > 30
                - {'country': ['UK', 'US']}: Generate samples from UK or US
                Currently supports exact matches only.
                
        Returns:
            pd.DataFrame: 
                A DataFrame containing n_samples rows of synthetic data.
                The DataFrame has the same columns as the original data,
                in the same order, with the same data types.
                
        Raises:
            ValueError: If n_samples is not positive or conditions are invalid.
            RuntimeError: If the generator has not been fitted.
            
        Example:
            >>> # Generate 1000 samples
            >>> synthetic = forge.generate(n_samples=1000)
            
            >>> # Generate with conditions
            >>> females = forge.generate(
            ...     n_samples=500,
            ...     conditions={'gender': 'female'}
            ... )
        """
        # STEP 1: VALIDATION
        
        # Check that the generator has been fitted
        if not self._is_fitted:
            raise RuntimeError(
                "TabularForge has not been fitted. This should not happen - "
                "please report this as a bug."
            )
        
        # Validate n_samples
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(
                f"n_samples must be a positive integer, got {n_samples}"
            )
        
        # STEP 2: GENERATE RAW SYNTHETIC DATA
        
        if self._verbose:
            logger.info(f"Generating {n_samples} synthetic samples...")
        
        # Generate samples using the fitted generator
        synthetic_transformed = self.generator.sample(n_samples, conditions)
        
        # STEP 3: APPLY DIFFERENTIAL PRIVACY (OPTIONAL)
        
        if self.privacy is not None:
            if self._verbose:
                logger.info("Applying differential privacy...")
            synthetic_transformed = self.privacy.add_noise(synthetic_transformed)
        
        # STEP 4: INVERSE TRANSFORM TO ORIGINAL SPACE
        
        # Convert back from transformed space to original data format
        synthetic_data = self.transformer.inverse_transform(
            synthetic_transformed,
            self.encoder
        )
        
        # STEP 5: POST-PROCESSING
        
        # Ensure column order matches original data
        synthetic_data = synthetic_data[self._column_names]
        
        # Reset index to start from 0
        synthetic_data = synthetic_data.reset_index(drop=True)
        
        if self._verbose:
            logger.info(f"Successfully generated {len(synthetic_data)} samples!")
        
        return synthetic_data
    
    def evaluate_quality(
        self,
        synthetic_data: Optional[pd.DataFrame] = None,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate the quality of synthetic data.
        
        This method computes various metrics to assess how well the synthetic
        data matches the statistical properties of the original data.
        
        Args:
            synthetic_data (pd.DataFrame, optional): 
                Synthetic data to evaluate. If None, will generate new samples.
                
            n_samples (int, optional): 
                Number of samples to generate if synthetic_data is None.
                Defaults to 1000.
                
        Returns:
            Dict[str, float]: 
                Dictionary containing quality metrics:
                - 'statistical_similarity': Overall statistical similarity (0-1)
                - 'column_correlations': How well column correlations are preserved
                - 'distribution_match': How well distributions are matched
                - 'ml_utility': ML model performance on synthetic vs real data
                
        Example:
            >>> quality = forge.evaluate_quality()
            >>> print(f"Statistical similarity: {quality['statistical_similarity']:.2%}")
        """
        # Generate synthetic data if not provided
        if synthetic_data is None:
            synthetic_data = self.generate(n_samples=n_samples)
        
        # Calculate statistical metrics
        stat_metrics = StatisticalMetrics()
        statistical_scores = stat_metrics.compute(self.data, synthetic_data, self.encoder)
        
        # Calculate utility metrics
        util_metrics = UtilityMetrics()
        utility_scores = util_metrics.compute(self.data, synthetic_data, self.encoder)
        
        # Combine all metrics
        quality_report = {
            "statistical_similarity": statistical_scores["overall"],
            "column_correlations": statistical_scores["correlation"],
            "distribution_match": statistical_scores["distribution"],
            "ml_utility": utility_scores["overall"]
        }
        
        return quality_report
    
    def evaluate_privacy(
        self,
        synthetic_data: Optional[pd.DataFrame] = None,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate privacy protection of synthetic data.
        
        This method tests the synthetic data against common privacy attacks
        to assess how well individual records are protected.
        
        Args:
            synthetic_data (pd.DataFrame, optional): 
                Synthetic data to evaluate. If None, will generate new samples.
                
            n_samples (int, optional): 
                Number of samples to generate if synthetic_data is None.
                
        Returns:
            Dict[str, float]: 
                Dictionary containing privacy metrics:
                - 'membership_inference_risk': Risk of membership inference attack
                - 'attribute_inference_risk': Risk of attribute inference attack  
                - 'nearest_neighbor_distance': Average distance to nearest real record
                
        Example:
            >>> privacy = forge.evaluate_privacy()
            >>> print(f"Membership inference risk: {privacy['membership_inference_risk']:.2%}")
        """
        # Generate synthetic data if not provided
        if synthetic_data is None:
            synthetic_data = self.generate(n_samples=n_samples)
        
        # Calculate privacy metrics
        priv_metrics = PrivacyMetrics()
        privacy_scores = priv_metrics.compute(self.data, synthetic_data, self.encoder)
        
        return privacy_scores
    
    def benchmark(
        self,
        generators: Optional[List[str]] = None,
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Benchmark multiple generators on this data.
        
        This method trains and evaluates multiple generators to help you
        choose the best one for your specific data.
        
        Args:
            generators (List[str], optional): 
                List of generator names to benchmark. If None, benchmarks all
                available generators.
                
            n_samples (int, optional): 
                Number of samples to generate for evaluation. Defaults to 1000.
                
        Returns:
            pd.DataFrame: 
                Benchmark results with generators as rows and metrics as columns.
                
        Example:
            >>> results = forge.benchmark(generators=['copula', 'ctgan'])
            >>> print(results)
        """
        # Default to all generators
        if generators is None:
            generators = list(AVAILABLE_GENERATORS.keys())
        
        results = []
        
        for gen_name in generators:
            if self._verbose:
                logger.info(f"Benchmarking {gen_name}...")
            
            try:
                # Create a new TabularForge with this generator
                temp_forge = TabularForge(
                    self.data,
                    generator=gen_name,
                    categorical_columns=self.encoder.categorical_columns,
                    numerical_columns=self.encoder.numerical_columns,
                    privacy_epsilon=self._privacy_epsilon,
                    random_state=self._random_state,
                    verbose=False
                )
                
                # Generate and evaluate
                synthetic = temp_forge.generate(n_samples=n_samples)
                quality = temp_forge.evaluate_quality(synthetic)
                privacy = temp_forge.evaluate_privacy(synthetic)
                
                # Store results
                results.append({
                    "generator": gen_name,
                    **quality,
                    **privacy
                })
                
            except Exception as e:
                logger.warning(f"Failed to benchmark {gen_name}: {e}")
                results.append({
                    "generator": gen_name,
                    "error": str(e)
                })
        
        return pd.DataFrame(results)
    
    def __repr__(self) -> str:
        """Return string representation of TabularForge instance."""
        return (
            f"TabularForge("
            f"n_samples={self._n_original_samples}, "
            f"n_columns={len(self._column_names)}, "
            f"generator='{self._generator_name}', "
            f"privacy_epsilon={self._privacy_epsilon})"
        )
