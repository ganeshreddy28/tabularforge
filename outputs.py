"""
Generate outputs for research paper
"""
import pandas as pd
import numpy as np
import json
from tabularforge import TabularForge

# Create realistic dataset
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'age': np.random.normal(45, 15, n).clip(18, 85).astype(int),
    'income': np.random.lognormal(10.5, 0.8, n).astype(int),
    'credit_score': np.random.normal(700, 80, n).clip(300, 850).astype(int),
    'gender': np.random.choice(['male', 'female'], n),
    'education': np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], n, p=[0.3, 0.4, 0.2, 0.1]),
    'employment': np.random.choice(['employed', 'self_employed', 'unemployed', 'retired'], n, p=[0.6, 0.15, 0.1, 0.15])
})

results = {}

# Test each generator
for generator in ['copula', 'ctgan', 'tvae']:
    print(f"\nTesting {generator}...")
    
    forge = TabularForge(data, generator=generator, random_state=42)
    synthetic = forge.generate(n_samples=1000)
    
    quality = forge.evaluate_quality(synthetic)
    privacy = forge.evaluate_privacy(synthetic)
    
    results[generator] = {
        'quality': quality,
        'privacy': privacy
    }
    
    # Save synthetic data
    synthetic.to_csv(f'synthetic_{generator}.csv', index=False)

# Save results
with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save original data
data.to_csv('original_data.csv', index=False)

print("\n✅ Saved:")
print("  - original_data.csv")
print("  - synthetic_copula.csv")
print("  - synthetic_ctgan.csv")
print("  - synthetic_tvae.csv")
print("  - benchmark_results.json")