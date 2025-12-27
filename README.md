# 🔨 TabularForge

<p align="center">
  <img src="docs/logo.png" alt="TabularForge Logo" width="200"/>
</p>

<p align="center">
  <strong>Privacy-Preserving Synthetic Tabular Data Generation</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/tabularforge-sgk/0.1.0/"><img src="https://img.shields.io/pypi/v/tabularforge.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/tabularforge-sgk/0.1.0/"><img src="https://img.shields.io/pypi/pyversions/tabularforge.svg" alt="Python versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/ganeshreddy28/tabularforge/actions"><img src="https://github.com/ganeshreddy28/tabularforge/workflows/tests/badge.svg" alt="Tests"></a>
</p>

---

## 🎯 What is TabularForge?

**TabularForge** is a unified, production-ready Python library for generating high-quality synthetic tabular data with built-in privacy guarantees. It combines multiple state-of-the-art approaches (GANs, VAEs, Copulas) into a simple, one-line API.

### Why Synthetic Data?

Organizations have valuable tabular data (patient records, financial transactions, customer data) but often can't share it due to:
- **Privacy regulations** (GDPR, HIPAA, CCPA)
- **Competitive sensitivity**
- **Data scarcity** for ML development

Synthetic data solves this by generating realistic, statistically similar data that protects individual privacy while preserving analytical utility.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Multiple Generators** | CTGAN, TVAE, Gaussian Copula, and more |
| 🔒 **Differential Privacy** | Formal privacy guarantees with configurable epsilon |
| 📊 **Quality Metrics** | Statistical similarity, ML utility, privacy leakage tests |
| 🔧 **Auto Preprocessing** | Handles mixed types, missing values, imbalanced data |
| ⚡ **One-Line API** | Generate synthetic data in a single line of code |
| 📈 **Benchmarking** | Compare generators on your specific data |

---

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install tabularforge

# Or install from source
git clone https://github.com/ganeshreddy28/tabularforge.git
cd tabularforge
pip install -e .
```

### Basic Usage

```python
from tabularforge import TabularForge
import pandas as pd

# Load your real data
real_data = pd.read_csv("your_data.csv")

# Generate synthetic data in ONE line!
forge = TabularForge(real_data)
synthetic_data = forge.generate(n_samples=1000)

# That's it! synthetic_data is a pandas DataFrame
print(synthetic_data.head())
```

### With Privacy Guarantees

```python
from tabularforge import TabularForge

# Generate with differential privacy (epsilon=1.0)
forge = TabularForge(real_data, privacy_epsilon=1.0)
private_synthetic = forge.generate(n_samples=1000)

# Check privacy metrics
privacy_report = forge.evaluate_privacy()
print(privacy_report)
```

### Compare Different Generators

```python
from tabularforge import TabularForge

# Benchmark all available generators
forge = TabularForge(real_data)
benchmark_results = forge.benchmark(generators=['ctgan', 'tvae', 'copula'])

# See which generator works best for your data
print(benchmark_results)
```

---

## 📖 Detailed Usage

### Choosing a Generator

TabularForge supports multiple synthetic data generators:

| Generator | Best For | Speed | Quality |
|-----------|----------|-------|---------|
| `copula` | Simple distributions, fast generation | ⚡⚡⚡ | ⭐⭐⭐ |
| `ctgan` | Complex relationships, mixed types | ⚡⚡ | ⭐⭐⭐⭐ |
| `tvae` | High-dimensional data | ⚡⚡ | ⭐⭐⭐⭐ |

```python
# Specify a generator
forge = TabularForge(real_data, generator='ctgan')
synthetic = forge.generate(n_samples=500)
```

### Handling Different Data Types

TabularForge automatically detects and handles:
- **Numerical columns** (continuous and discrete)
- **Categorical columns** (including high-cardinality)
- **DateTime columns**
- **Missing values**

```python
# Explicit column type specification (optional)
forge = TabularForge(
    real_data,
    categorical_columns=['gender', 'country', 'product_type'],
    numerical_columns=['age', 'income', 'score'],
    datetime_columns=['signup_date', 'last_purchase']
)
```

### Evaluating Synthetic Data Quality

```python
from tabularforge import TabularForge

forge = TabularForge(real_data)
synthetic = forge.generate(n_samples=1000)

# Get comprehensive quality report
quality_report = forge.evaluate_quality(synthetic)

print(quality_report)
# Output:
# {
#     'statistical_similarity': 0.92,
#     'column_correlations': 0.89,
#     'distribution_match': 0.94,
#     'ml_utility': 0.87
# }
```

### Conditional Generation

Generate data satisfying specific conditions:

```python
# Generate only high-income customers
synthetic = forge.generate(
    n_samples=500,
    conditions={'income': '>100000', 'country': 'UK'}
)
```

---

## 🔒 Privacy Features

### Differential Privacy

TabularForge implements differential privacy to provide formal privacy guarantees:

```python
# Lower epsilon = stronger privacy (but lower utility)
# Higher epsilon = weaker privacy (but higher utility)
forge = TabularForge(real_data, privacy_epsilon=0.1)  # Strong privacy
forge = TabularForge(real_data, privacy_epsilon=1.0)  # Balanced
forge = TabularForge(real_data, privacy_epsilon=10.0) # Weak privacy
```

### Privacy Attack Simulation

Test your synthetic data against common privacy attacks:

```python
# Simulate membership inference attack
attack_results = forge.simulate_attack(
    attack_type='membership_inference',
    synthetic_data=synthetic
)

print(f"Attack success rate: {attack_results['success_rate']:.2%}")
# A good synthetic dataset should have ~50% (random guess)
```

---

## 📊 Use Cases

### Healthcare
```python
# Generate synthetic patient cohorts for research
patient_data = pd.read_csv("patient_records.csv")
forge = TabularForge(patient_data, privacy_epsilon=1.0)
synthetic_patients = forge.generate(n_samples=10000)
# Share with researchers without exposing real patients
```

### Finance
```python
# Create synthetic transactions for fraud detection R&D
transactions = pd.read_csv("transactions.csv")
forge = TabularForge(transactions)
synthetic_transactions = forge.generate(n_samples=50000)
# Develop ML models without sensitive financial data
```

### ML Development
```python
# Augment small datasets
small_dataset = pd.read_csv("rare_events.csv")  # Only 100 samples
forge = TabularForge(small_dataset)
augmented = forge.generate(n_samples=10000)
# Now you have enough data to train robust models
```

---

## 🏗️ Architecture

```
tabularforge/
├── __init__.py              # Main API exports
├── forge.py                 # TabularForge main class
├── generators/              # Synthetic data generators
│   ├── base.py              # Abstract base generator
│   ├── copula.py            # Gaussian Copula generator
│   ├── ctgan.py             # CTGAN generator
│   └── tvae.py              # TVAE generator
├── preprocessing/           # Data preprocessing
│   ├── encoder.py           # Column encoding/decoding
│   └── transformer.py       # Data transformations
├── privacy/                 # Privacy mechanisms
│   ├── differential.py      # Differential privacy
│   └── attacks.py           # Privacy attack simulations
├── metrics/                 # Quality & privacy metrics
│   ├── statistical.py       # Statistical similarity
│   ├── utility.py           # ML utility metrics
│   └── privacy.py           # Privacy metrics
└── utils/                   # Utilities
    ├── config.py            # Configuration management
    └── logging.py           # Logging utilities
```

---

## 🧪 Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/ganeshreddy28/tabularforge.git
cd tabularforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
flake8 tabularforge/
black tabularforge/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tabularforge --cov-report=html

# Run specific test file
pytest tests/test_generators.py -v
```

---

## 📚 Documentation

- [Full Documentation](https://tabularforge.readthedocs.io/)
- [API Reference](https://tabularforge.readthedocs.io/api/)
- [Tutorials](https://tabularforge.readthedocs.io/tutorials/)
- [Examples](https://github.com/yourusername/tabularforge/tree/main/examples)

---

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [SDV](https://github.com/sdv-dev/SDV) for inspiration on synthetic data APIs
- [CTGAN Paper](https://arxiv.org/abs/1907.00503) for the CTGAN architecture
- The differential privacy research community

---

## 📬 Contact

- **Author**: Sai Ganesh Kolan
- **Email**: aiganesh1299@gmail.com
- **LinkedIn**: (https://linkedin.com/in/saiganeshkolan/)

---

<p align="center">
  Made with ❤️ for the data science community
</p>

<p align="center">
  ⭐ Star us on GitHub if you find this useful!
</p>
