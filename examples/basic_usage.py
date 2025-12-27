"""
TabularForge Examples
---------------------

This file contains examples demonstrating how to use TabularForge
for synthetic tabular data generation.

Run this file:
    python examples/basic_usage.py

Author: Sai Ganesh Kolan
License: MIT
"""

import numpy as np
import pandas as pd
from tabularforge import TabularForge


def create_sample_data():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic sample data
    data = pd.DataFrame({
        # Demographics
        "age": np.random.normal(35, 12, n_samples).clip(18, 75).astype(int),
        "gender": np.random.choice(["male", "female", "other"], n_samples, p=[0.48, 0.48, 0.04]),
        "country": np.random.choice(["UK", "US", "Germany", "France", "Spain"], n_samples),
        
        # Financial
        "income": np.random.lognormal(10.5, 0.8, n_samples).astype(int),
        "credit_score": np.random.normal(700, 80, n_samples).clip(300, 850).astype(int),
        
        # Behavioral
        "purchase_frequency": np.random.poisson(5, n_samples),
        "loyalty_score": np.random.uniform(0, 100, n_samples).round(2),
        
        # Categorical
        "membership_tier": np.random.choice(
            ["bronze", "silver", "gold", "platinum"], 
            n_samples, 
            p=[0.5, 0.3, 0.15, 0.05]
        ),
    })
    
    return data


def example_1_basic_usage():
    """
    Example 1: Basic Synthetic Data Generation
    -------------------------------------------
    The simplest way to generate synthetic data.
    """
    print("\n" + "-"*60)
    print("Example 1: Basic Synthetic Data Generation")
    print("-"*60)
    
    # Create sample data
    real_data = create_sample_data()
    print(f"\nOriginal data shape: {real_data.shape}")
    print(f"Original data sample:\n{real_data.head()}")
    
    # Generate synthetic data in ONE line!
    forge = TabularForge(real_data)
    synthetic_data = forge.generate(n_samples=500)
    
    print(f"\nSynthetic data shape: {synthetic_data.shape}")
    print(f"Synthetic data sample:\n{synthetic_data.head()}")
    
    return synthetic_data


def example_2_with_privacy():
    """
    Example 2: Synthetic Data with Differential Privacy
    ----------------------------------------------------
    Add formal privacy guarantees to your synthetic data.
    """
    print("\n" + "-"*60)
    print("Example 2: Synthetic Data with Differential Privacy")
    print("-"*60)
    
    real_data = create_sample_data()
    
    # Generate with differential privacy (epsilon=1.0 is a common choice)
    forge = TabularForge(
        real_data,
        privacy_epsilon=1.0,  # Lower = more privacy, more noise
        random_state=42       # For reproducibility
    )
    
    private_synthetic = forge.generate(n_samples=500)
    
    print(f"\nGenerated {len(private_synthetic)} private synthetic samples")
    print(f"Privacy epsilon: {forge._privacy_epsilon}")
    print(f"\nSample:\n{private_synthetic.head()}")
    
    return private_synthetic


def example_3_quality_evaluation():
    """
    Example 3: Evaluating Synthetic Data Quality
    ---------------------------------------------
    Check how well synthetic data matches the original.
    """
    print("\n" + "-"*60)
    print("Example 3: Evaluating Synthetic Data Quality")
    print("-"*60)
    
    real_data = create_sample_data()
    
    forge = TabularForge(real_data)
    synthetic_data = forge.generate(n_samples=1000)
    
    # Evaluate quality
    quality = forge.evaluate_quality(synthetic_data)
    
    print("\nQuality Metrics:")
    print("-" * 40)
    for metric, score in quality.items():
        print(f"  {metric}: {score:.2%}")
    
    # Evaluate privacy
    privacy = forge.evaluate_privacy(synthetic_data)
    
    print("\nPrivacy Metrics:")
    print("-" * 40)
    for metric, score in privacy.items():
        if isinstance(score, float):
            print(f"  {metric}: {score:.4f}")
        else:
            print(f"  {metric}: {score}")
    
    return quality, privacy


def example_4_different_generators():
    """
    Example 4: Comparing Different Generators
    ------------------------------------------
    TabularForge supports multiple generation algorithms.
    """
    print("\n" + "-"*60)
    print("Example 4: Comparing Different Generators")
    print("-"*60)
    
    real_data = create_sample_data()
    
    generators = ["copula", "ctgan", "tvae"]
    
    for gen_name in generators:
        print(f"\nUsing {gen_name.upper()} generator...")
        
        forge = TabularForge(
            real_data,
            generator=gen_name,
            random_state=42
        )
        
        synthetic = forge.generate(n_samples=200)
        quality = forge.evaluate_quality(synthetic)
        
        print(f"  Quality: {quality['statistical_similarity']:.2%}")
        print(f"  Sample:\n{synthetic.head(3)}")


def example_5_column_specification():
    """
    Example 5: Explicit Column Type Specification
    -----------------------------------------------
    Sometimes you want to explicitly specify column types.
    """
    print("\n" + "-"*60)
    print("Example 5: Explicit Column Type Specification")
    print("-"*60)
    
    real_data = create_sample_data()
    
    forge = TabularForge(
        real_data,
        categorical_columns=["gender", "country", "membership_tier"],
        numerical_columns=["age", "income", "credit_score", "purchase_frequency", "loyalty_score"],
    )
    
    print("\nEncoder detected:")
    print(f"  Categorical: {forge.encoder.categorical_columns}")
    print(f"  Numerical: {forge.encoder.numerical_columns}")
    
    synthetic = forge.generate(n_samples=500)
    print(f"\nGenerated {len(synthetic)} samples")


def example_6_healthcare_use_case():
    """
    Example 6: Healthcare Use Case
    --------------------------------
    Generate synthetic patient data for research.
    """
    print("\n" + "-"*60)
    print("Example 6: Healthcare Use Case")
    print("-"*60)
    
    # Create mock patient data
    np.random.seed(42)
    n_patients = 500
    
    patient_data = pd.DataFrame({
        "patient_age": np.random.normal(55, 15, n_patients).clip(18, 90).astype(int),
        "bmi": np.random.normal(26, 5, n_patients).clip(15, 45).round(1),
        "blood_pressure_systolic": np.random.normal(125, 15, n_patients).clip(90, 180).astype(int),
        "blood_pressure_diastolic": np.random.normal(80, 10, n_patients).clip(60, 120).astype(int),
        "cholesterol": np.random.choice(["normal", "high", "very_high"], n_patients, p=[0.6, 0.3, 0.1]),
        "diabetes": np.random.choice(["no", "type1", "type2"], n_patients, p=[0.85, 0.05, 0.1]),
        "smoker": np.random.choice(["never", "former", "current"], n_patients, p=[0.5, 0.3, 0.2]),
    })
    
    print(f"Original patient records: {len(patient_data)}")
    
    # Generate synthetic patients with strong privacy
    forge = TabularForge(
        patient_data,
        privacy_epsilon=0.5,  # Strong privacy for healthcare
        random_state=42
    )
    
    synthetic_patients = forge.generate(n_samples=1000)
    
    print(f"Generated synthetic patients: {len(synthetic_patients)}")
    print(f"\nSynthetic patient sample:\n{synthetic_patients.head()}")
    
    # Evaluate
    quality = forge.evaluate_quality(synthetic_patients)
    print(f"\nQuality score: {quality['statistical_similarity']:.2%}")
    
    return synthetic_patients


if __name__ == "__main__":
    print("\n" + "*"*60)
    print("# TabularForge Examples")
    print("*"*60)
    
    # Run all examples
    example_1_basic_usage()
    example_2_with_privacy()
    example_3_quality_evaluation()
    example_4_different_generators()
    example_5_column_specification()
    example_6_healthcare_use_case()
    
    print("\n" + "-"*60)
    print("All examples completed successfully!")
    print("-"*60)
