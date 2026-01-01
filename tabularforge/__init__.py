"""
TabularForge: Privacy-Preserving Synthetic Tabular Data Generation

TabularForge is a unified, production-ready Python library for generating
high-quality synthetic tabular data with built-in privacy guarantees.

Basic Usage
-----------
>>> from tabularforge import TabularForge
>>> import pandas as pd
>>>
>>> # Load your real data
>>> real_data = pd.read_csv("your_data.csv")
>>>
>>> # Generate synthetic data in ONE line!
>>> forge = TabularForge(real_data)
>>> synthetic_data = forge.generate(n_samples=1000)

With Privacy
------------
>>> # Generate with differential privacy
>>> forge = TabularForge(real_data, privacy_epsilon=1.0)
>>> private_synthetic = forge.generate(n_samples=1000)

For more information, see:
- Documentation: https://tabularforge.readthedocs.io
- GitHub: https://github.com/yourusername/tabularforge

Author: Sai Ganesh Kolan
License: MIT
"""

# VERSION INFORMATION
# We follow Semantic Versioning (SemVer): MAJOR.MINOR.PATCH
# - MAJOR: Incompatible API changes
# - MINOR: New functionality (backwards compatible)
# - PATCH: Bug fixes (backwards compatible)
__version__ = "0.1.1"
__author__ = "Sai Ganesh Kolan"
__email__ = "aiganesh1299@gmail.com"
__license__ = "MIT"

# PUBLIC API IMPORTS
# These are the main classes and functions users should use.
# We import them here so users can do: from tabularforge import TabularForge

# Main class - the primary interface for the library
from tabularforge.forge import TabularForge

# Generator classes - for users who want more control
from tabularforge.generators.base import BaseGenerator
from tabularforge.generators.copula import GaussianCopulaGenerator
from tabularforge.generators.ctgan import CTGANGenerator
from tabularforge.generators.tvae import TVAEGenerator

# Preprocessing utilities
from tabularforge.preprocessing.encoder import DataEncoder
from tabularforge.preprocessing.transformer import DataTransformer

# Privacy mechanisms
from tabularforge.privacy.differential import DifferentialPrivacy
from tabularforge.privacy.attacks import PrivacyAttackSimulator

# Metrics for evaluation
from tabularforge.metrics.statistical import StatisticalMetrics
from tabularforge.metrics.utility import UtilityMetrics
from tabularforge.metrics.privacy import PrivacyMetrics


# __all__ DEFINITION
# This list defines what gets exported when someone does:
# from tabularforge import *
# We explicitly list everything to make the public API clear.
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Main class
    "TabularForge",
    
    # Generators
    "BaseGenerator",
    "GaussianCopulaGenerator", 
    "CTGANGenerator",
    "TVAEGenerator",
    
    # Preprocessing
    "DataEncoder",
    "DataTransformer",
    
    # Privacy
    "DifferentialPrivacy",
    "PrivacyAttackSimulator",
    
    # Metrics
    "StatisticalMetrics",
    "UtilityMetrics",
    "PrivacyMetrics",
]

# PACKAGE INITIALISATION LOGGING
# Optional: Log package import for debugging purposes
# This can be helpful during development but should be quiet in production

import logging

# Create a logger for the package
logger = logging.getLogger(__name__)

# Set default handler to avoid "No handlers could be found" warning
logger.addHandler(logging.NullHandler())

# Log package import at DEBUG level (won't show unless user configures logging)
logger.debug(f"TabularForge v{__version__} initialized")
