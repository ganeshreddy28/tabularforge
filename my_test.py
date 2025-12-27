"""
My TabularForge Test Script
"""
import pandas as pd
import numpy as np
from tabularforge import TabularForge

# Create sample data
print("Creating sample data...")
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(35, 10, 1000).clip(18, 75).astype(int),
    'income': np.random.lognormal(10, 1, 1000).astype(int),
    'gender': np.random.choice(['M', 'F'], 1000),
    'country': np.random.choice(['UK', 'US', 'Germany'], 1000),
    'score': np.random.uniform(0, 100, 1000).round(2)
})

print(f"Original data shape: {data.shape}")
print(f"\nOriginal data sample:\n{data.head()}")

# Generate synthetic data
print("\n" + "="*50)
print("Generating synthetic data...")
print("="*50)

forge = TabularForge(data, generator='copula', random_state=42)
synthetic = forge.generate(n_samples=500)

print(f"\nSynthetic data shape: {synthetic.shape}")
print(f"\nSynthetic data sample:\n{synthetic.head()}")

# Evaluate quality
print("\n" + "="*50)
print("Evaluating quality...")
print("="*50)

quality = forge.evaluate_quality(synthetic)
print("\nQuality Metrics:")
for metric, score in quality.items():
    print(f"  {metric}: {score:.2%}")

# Evaluate privacy
print("\n" + "="*50)
print("Evaluating privacy...")
print("="*50)

privacy = forge.evaluate_privacy(synthetic)
print("\nPrivacy Metrics:")
for metric, value in privacy.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")

# Test with privacy
print("\n" + "="*50)
print("Testing with differential privacy (epsilon=1.0)...")
print("="*50)

forge_private = TabularForge(data, privacy_epsilon=1.0, random_state=42)
private_synthetic = forge_private.generate(n_samples=500)
print(f"\nPrivate synthetic sample:\n{private_synthetic.head()}")

print("\n" + "="*50)
print("✅ All tests completed successfully!")
print("="*50)