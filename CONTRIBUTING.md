# Contributing to TabularForge

Thank you for your interest in contributing to TabularForge! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

Please be respectful and constructive in all interactions. We're building something useful together!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tabularforge.git
   cd tabularforge
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/tabularforge.git
   ```

## Development Setup

### 1. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Install Pre-commit Hooks (Optional but Recommended)

```bash
pre-commit install
```

## Making Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, commented code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tabularforge --cov-report=html
```

### 4. Check Code Style

```bash
# Format code
black tabularforge/
isort tabularforge/

# Check linting
flake8 tabularforge/
```

## Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub

3. **Describe your changes**:
   - What does this PR do?
   - Why is this change needed?
   - How was it tested?

4. **Wait for review** and address any feedback

## Code Style

We follow these conventions:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **Type hints** for function signatures
- **Docstrings** for all public functions and classes (Google style)

Example:

```python
def generate_synthetic_data(
    real_data: pd.DataFrame,
    n_samples: int = 1000,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic tabular data.
    
    Args:
        real_data: The original data to learn from.
        n_samples: Number of synthetic samples to generate.
        random_state: Seed for reproducibility.
        
    Returns:
        DataFrame containing synthetic samples.
        
    Raises:
        ValueError: If real_data is empty.
    """
    ...
```

## Testing

- **All new features** must have tests
- **Bug fixes** should include a test that would have caught the bug
- Tests should be in the `tests/` directory
- Use pytest fixtures for reusable test data

Example test:

```python
def test_generate_returns_correct_shape(sample_data):
    """Test that generate() returns the expected number of samples."""
    forge = TabularForge(sample_data)
    synthetic = forge.generate(n_samples=100)
    
    assert len(synthetic) == 100
    assert list(synthetic.columns) == list(sample_data.columns)
```

## Documentation

- Update the README if you add new features
- Add docstrings to all public functions
- Include examples where helpful

## Questions?

Feel free to open an issue if you have questions or need help!

---

Thank you for contributing to TabularForge! 🎉
