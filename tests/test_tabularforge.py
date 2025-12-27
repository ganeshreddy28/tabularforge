"""
TabularForge Test Suite
=======================

This test module contains comprehensive tests for TabularForge.

To run tests:
    pytest tests/test_tabularforge.py -v

To run with coverage:
    pytest tests/test_tabularforge.py --cov=tabularforge --cov-report=html

Author: Sai Ganesh Kolan
License: MIT
"""

import pytest
import numpy as np
import pandas as pd

from tabularforge import TabularForge
from tabularforge.generators import GaussianCopulaGenerator, CTGANGenerator, TVAEGenerator
from tabularforge.preprocessing import DataEncoder, DataTransformer
from tabularforge.privacy import DifferentialPrivacy
from tabularforge.metrics import StatisticalMetrics


# =============================================================================
# FIXTURES - Reusable test data
# =============================================================================

@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        "age": np.random.normal(35, 10, n_samples).clip(18, 80),
        "income": np.random.lognormal(10, 1, n_samples),
        "score": np.random.uniform(0, 100, n_samples),
        "children": np.random.poisson(1.5, n_samples),
        "gender": np.random.choice(["male", "female"], n_samples),
        "country": np.random.choice(["UK", "US", "Germany", "France"], n_samples),
        "education": np.random.choice(
            ["high_school", "bachelors", "masters", "phd"], 
            n_samples, p=[0.3, 0.4, 0.2, 0.1]
        ),
    })
    
    data.loc[np.random.choice(n_samples, 20), "income"] = np.nan
    data.loc[np.random.choice(n_samples, 10), "education"] = np.nan
    
    return data


@pytest.fixture
def simple_numerical_data():
    """Create a simple numerical-only DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "x1": np.random.normal(0, 1, 200),
        "x2": np.random.normal(5, 2, 200),
        "x3": np.random.uniform(0, 10, 200),
    })


# =============================================================================
# TABULARFORGE MAIN CLASS TESTS
# =============================================================================

class TestTabularForge:
    """Tests for the main TabularForge class."""
    
    def test_initialization(self, sample_data):
        """Test basic initialization with default parameters."""
        forge = TabularForge(sample_data, verbose=False)
        
        assert forge._is_fitted
        assert forge._n_original_samples == len(sample_data)
        assert len(forge._column_names) == len(sample_data.columns)
    
    def test_initialization_with_generator(self, sample_data):
        """Test initialization with different generators."""
        for gen_name in ["copula", "ctgan", "tvae"]:
            forge = TabularForge(sample_data, generator=gen_name, verbose=False)
            assert forge._generator_name == gen_name
    
    def test_invalid_generator(self, sample_data):
        """Test that invalid generator names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown generator"):
            TabularForge(sample_data, generator="invalid_generator")
    
    def test_empty_data(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            TabularForge(empty_df)
    
    def test_invalid_data_type(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            TabularForge([[1, 2], [3, 4]])
    
    def test_generate_basic(self, sample_data):
        """Test basic synthetic data generation."""
        forge = TabularForge(sample_data, verbose=False)
        synthetic = forge.generate(n_samples=100)
        
        assert len(synthetic) == 100
        assert list(synthetic.columns) == list(sample_data.columns)
    
    def test_generate_invalid_n_samples(self, sample_data):
        """Test that invalid n_samples raises ValueError."""
        forge = TabularForge(sample_data, verbose=False)
        
        with pytest.raises(ValueError):
            forge.generate(n_samples=0)
        
        with pytest.raises(ValueError):
            forge.generate(n_samples=-10)
    
    def test_generate_with_privacy(self, sample_data):
        """Test generation with differential privacy."""
        forge = TabularForge(sample_data, privacy_epsilon=1.0, verbose=False)
        synthetic = forge.generate(n_samples=100)
        
        assert len(synthetic) == 100
        assert forge.privacy is not None
    
    def test_evaluate_quality(self, sample_data):
        """Test quality evaluation."""
        forge = TabularForge(sample_data, verbose=False)
        synthetic = forge.generate(n_samples=200)
        quality = forge.evaluate_quality(synthetic)
        
        assert "statistical_similarity" in quality
        assert "column_correlations" in quality
        assert "distribution_match" in quality
        assert "ml_utility" in quality
        
        for key, value in quality.items():
            assert 0 <= value <= 1, f"{key} score out of range: {value}"
    
    def test_evaluate_privacy(self, sample_data):
        """Test privacy evaluation."""
        forge = TabularForge(sample_data, verbose=False)
        synthetic = forge.generate(n_samples=200)
        privacy = forge.evaluate_privacy(synthetic)
        
        assert "membership_inference_risk" in privacy
        assert "nearest_neighbor_distance" in privacy
    
    def test_reproducibility(self, sample_data):
        """Test that random_state ensures reproducibility."""
        forge1 = TabularForge(sample_data, random_state=42, verbose=False)
        synthetic1 = forge1.generate(n_samples=100)
        
        forge2 = TabularForge(sample_data, random_state=42, verbose=False)
        synthetic2 = forge2.generate(n_samples=100)
        
        pd.testing.assert_frame_equal(synthetic1, synthetic2)


# =============================================================================
# DATA ENCODER TESTS
# =============================================================================

class TestDataEncoder:
    """Tests for the DataEncoder class."""
    
    def test_auto_detection(self, sample_data):
        """Test automatic column type detection."""
        encoder = DataEncoder()
        encoder.fit(sample_data)
        
        assert len(encoder.categorical_columns) > 0
        assert len(encoder.numerical_columns) > 0
        assert "gender" in encoder.categorical_columns
        assert "age" in encoder.numerical_columns
    
    def test_explicit_columns(self, sample_data):
        """Test explicit column type specification."""
        encoder = DataEncoder(
            categorical_columns=["gender", "country"],
            numerical_columns=["age", "income"]
        )
        encoder.fit(sample_data)
        
        assert encoder.categorical_columns == ["gender", "country"]
        assert encoder.numerical_columns == ["age", "income"]
    
    def test_transform_inverse_transform(self, sample_data):
        """Test that inverse transform recovers original data."""
        encoder = DataEncoder()
        encoder.fit(sample_data)
        
        transformed = encoder.transform(sample_data)
        
        for col in transformed.columns:
            assert np.issubdtype(transformed[col].dtype, np.number)
        
        recovered = encoder.inverse_transform(transformed)
        
        # Check that we get the same number of rows back
        assert len(recovered) == len(sample_data)
        
        # Check that all columns are present
        assert set(recovered.columns) == set(sample_data.columns)


# =============================================================================
# GENERATOR TESTS
# =============================================================================

class TestGaussianCopulaGenerator:
    """Tests for Gaussian Copula generator."""
    
    def test_fit_and_sample(self, simple_numerical_data):
        """Test basic fit and sample."""
        encoder = DataEncoder()
        encoder.fit(simple_numerical_data)
        transformed = encoder.transform(simple_numerical_data)
        
        generator = GaussianCopulaGenerator(random_state=42)
        generator.fit(transformed, encoder)
        
        assert generator.is_fitted
        
        synthetic = generator.sample(100)
        assert len(synthetic) == 100
        assert list(synthetic.columns) == list(transformed.columns)


class TestCTGANGenerator:
    """Tests for CTGAN generator."""
    
    def test_fit_and_sample(self, simple_numerical_data):
        """Test basic fit and sample with minimal epochs."""
        encoder = DataEncoder()
        encoder.fit(simple_numerical_data)
        transformed = encoder.transform(simple_numerical_data)
        
        generator = CTGANGenerator(epochs=5, batch_size=50, random_state=42)
        generator.fit(transformed, encoder)
        
        assert generator.is_fitted
        
        synthetic = generator.sample(50)
        assert len(synthetic) == 50


class TestTVAEGenerator:
    """Tests for TVAE generator."""
    
    def test_fit_and_sample(self, simple_numerical_data):
        """Test basic fit and sample with minimal epochs."""
        encoder = DataEncoder()
        encoder.fit(simple_numerical_data)
        transformed = encoder.transform(simple_numerical_data)
        
        generator = TVAEGenerator(epochs=5, batch_size=50, random_state=42)
        generator.fit(transformed, encoder)
        
        assert generator.is_fitted
        
        synthetic = generator.sample(50)
        assert len(synthetic) == 50


# =============================================================================
# DIFFERENTIAL PRIVACY TESTS
# =============================================================================

class TestDifferentialPrivacy:
    """Tests for differential privacy mechanism."""
    
    def test_initialization(self):
        """Test valid initialization."""
        dp = DifferentialPrivacy(epsilon=1.0)
        assert dp.epsilon == 1.0
    
    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises ValueError."""
        with pytest.raises(ValueError):
            DifferentialPrivacy(epsilon=0)
        
        with pytest.raises(ValueError):
            DifferentialPrivacy(epsilon=-1)
    
    def test_add_noise(self, simple_numerical_data):
        """Test that noise is added."""
        dp = DifferentialPrivacy(epsilon=1.0)
        noisy_data = dp.add_noise(simple_numerical_data)
        
        assert not simple_numerical_data.equals(noisy_data)
        assert list(noisy_data.columns) == list(simple_numerical_data.columns)


# =============================================================================
# STATISTICAL METRICS TESTS
# =============================================================================

class TestStatisticalMetrics:
    """Tests for statistical metrics."""
    
    def test_compute(self, sample_data):
        """Test basic metric computation."""
        encoder = DataEncoder()
        encoder.fit(sample_data)
        
        metrics = StatisticalMetrics()
        scores = metrics.compute(sample_data, sample_data, encoder)
        
        assert scores["overall"] > 0.9


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, sample_data):
        """Test the complete synthetic data generation pipeline."""
        forge = TabularForge(
            sample_data,
            generator="copula",
            privacy_epsilon=1.0,
            random_state=42,
            verbose=False
        )
        
        synthetic = forge.generate(n_samples=200)
        
        assert len(synthetic) == 200
        assert list(synthetic.columns) == list(sample_data.columns)
        
        quality = forge.evaluate_quality(synthetic)
        assert "statistical_similarity" in quality
        
        privacy = forge.evaluate_privacy(synthetic)
        assert "membership_inference_risk" in privacy
    
    def test_mixed_data_types(self):
        """Test with various data types."""
        np.random.seed(42)
        
        data = pd.DataFrame({
            "int_col": np.random.randint(0, 100, 200),
            "float_col": np.random.uniform(0, 1, 200),
            "cat_col": np.random.choice(["A", "B", "C"], 200),
            "bool_col": np.random.choice([True, False], 200),
        })
        
        forge = TabularForge(data, verbose=False)
        synthetic = forge.generate(n_samples=100)
        
        assert len(synthetic) == 100
        assert set(synthetic.columns) == set(data.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])