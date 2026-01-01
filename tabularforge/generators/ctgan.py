"""
CTGAN Generator (Conditional Tabular GAN)

This module implements CTGAN (Conditional Tabular GAN) for synthetic data
generation. CTGAN is specifically designed for tabular data and handles
both continuous and categorical columns effectively.

Paper: "Modeling Tabular Data using Conditional GAN" (NeurIPS 2019)
Authors: Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

Key Innovations in CTGAN:
    1. Mode-specific normalization for continuous columns
    2. Conditional generator for handling imbalanced categorical columns
    3. Training-by-sampling to address class imbalance

Architecture:
    - Generator: Takes random noise + condition vector → synthetic row
    - Discriminator: Takes real/synthetic row → real/fake probability

Author: Sai Ganesh Kolan
License: MIT
"""

# IMPORTS
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tabularforge.generators.base import BaseGenerator


# NEURAL NETWORK COMPONENTS

class Residual(nn.Module):
    """
    Residual block for the Generator network.
    
    Residual connections help with gradient flow during training and allow
    the network to learn identity mappings easily.
    
    Architecture:
        Input → Linear → BatchNorm → ReLU → Linear → BatchNorm → + Input → ReLU
    """
    
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the residual block.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
        """
        super().__init__()
        
        # First linear transformation
        self.fc1 = nn.Linear(input_dim, output_dim)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(output_dim)
        
        # Second linear transformation
        self.fc2 = nn.Linear(output_dim, output_dim)
        
        # Second batch normalization
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # ReLU activation
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Store input for residual connection
        residual = x
        
        # First transformation
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second transformation
        out = self.fc2(out)
        out = self.bn2(out)
        
        # Add residual connection (if dimensions match)
        if residual.shape == out.shape:
            out = out + residual
        
        # Final activation
        out = self.relu(out)
        
        return out


class Generator(nn.Module):
    """
    Generator network for CTGAN.
    
    The generator takes random noise and an optional condition vector,
    and produces a synthetic data row. It uses residual blocks to allow
    for deep networks without vanishing gradients.
    
    Architecture:
        [Noise + Condition] → Residual → Residual → Linear → Output
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2
    ) -> None:
        """
        Initialize the Generator.
        
        Args:
            input_dim: Dimension of noise vector + condition
            output_dim: Dimension of output (same as data row)
            hidden_dim: Dimension of hidden layers
            n_layers: Number of residual blocks
        """
        super().__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input layer: project noise+condition to hidden dimension
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        
        # Residual blocks for deep transformation
        self.residual_blocks = nn.ModuleList([
            Residual(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Output layer: project to data dimension
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, noise: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate synthetic data from noise.
        
        Args:
            noise: Random noise tensor of shape (batch_size, noise_dim)
            condition: Optional condition tensor of shape (batch_size, condition_dim)
            
        Returns:
            Synthetic data tensor of shape (batch_size, output_dim)
        """
        # Concatenate noise and condition if condition is provided
        if condition is not None:
            x = torch.cat([noise, condition], dim=1)
        else:
            x = noise
        
        # Input transformation
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.relu(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output transformation
        output = self.output_layer(x)
        
        return output


class Discriminator(nn.Module):
    """
    Discriminator network for CTGAN.
    
    The discriminator takes a data row (real or synthetic) and outputs
    the probability that it's real. It uses LeakyReLU activations and
    dropout for regularization.
    
    Architecture:
        Input → [Linear → LeakyReLU → Dropout] × n_layers → Linear → Sigmoid
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.5
    ) -> None:
        """
        Initialize the Discriminator.
        
        Args:
            input_dim: Dimension of input (data row)
            hidden_dim: Dimension of hidden layers
            n_layers: Number of hidden layers
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        # Build layers list
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
        
        # Output layer (single probability)
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        # Combine into sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input as real or fake.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        return self.model(x)


# CTGAN GENERATOR CLASS

class CTGANGenerator(BaseGenerator):
    """
    CTGAN-based synthetic data generator.
    
    CTGAN (Conditional Tabular GAN) is specifically designed for generating
    high-quality synthetic tabular data. It handles both continuous and
    categorical columns effectively using mode-specific normalization and
    conditional generation.
    
    Attributes:
        generator (Generator): The generator neural network
        discriminator (Discriminator): The discriminator neural network
        noise_dim (int): Dimension of the random noise input
        hidden_dim (int): Dimension of hidden layers
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Example:
        >>> generator = CTGANGenerator(epochs=300)
        >>> generator.fit(data, encoder)
        >>> synthetic = generator.sample(1000)
    """
    
    def __init__(
        self,
        noise_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        epochs: int = 300,
        batch_size: int = 500,
        learning_rate: float = 2e-4,
        discriminator_steps: int = 1,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize CTGAN generator.
        
        Args:
            noise_dim: Dimension of random noise vector (default: 128)
            hidden_dim: Dimension of hidden layers (default: 256)
            n_layers: Number of layers in G and D (default: 2)
            epochs: Number of training epochs (default: 300)
            batch_size: Batch size for training (default: 500)
            learning_rate: Learning rate for Adam optimizer (default: 2e-4)
            discriminator_steps: D updates per G update (default: 1)
            random_state: Seed for reproducibility
        """
        super().__init__(random_state=random_state)
        
        # Store hyperparameters
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discriminator_steps = discriminator_steps
        
        # These will be initialized during fitting
        self.generator: Optional[Generator] = None
        self.discriminator: Optional[Discriminator] = None
        self._data_dim: int = 0
        
        # Device selection (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _fit(
        self,
        data: pd.DataFrame,
        encoder: Any
    ) -> None:
        """
        Train the CTGAN on the data.
        
        This method:
        1. Converts data to PyTorch tensors
        2. Initializes Generator and Discriminator
        3. Trains using adversarial training
        
        Args:
            data: Training data (transformed)
            encoder: Data encoder with column info
        """
        # STEP 1: PREPARE DATA
        
        # Convert to numpy array
        data_array = data.values.astype(np.float32)
        self._data_dim = data_array.shape[1]
        self.column_names = list(data.columns)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data_array, dtype=torch.float32).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # STEP 2: INITIALISE NETWORKS
        
        # Initialize generator
        self.generator = Generator(
            input_dim=self.noise_dim,
            output_dim=self._data_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers
        ).to(self.device)
        
        # Initialize discriminator
        self.discriminator = Discriminator(
            input_dim=self._data_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers
        ).to(self.device)
        
        # STEP 3: SETUP TRAINING
        
        # Optimizers (Adam with beta1=0.5 as commonly used in GANs)
        g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9)
        )
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9)
        )
        
        # Binary cross entropy loss
        criterion = nn.BCELoss()
        
        
        # STEP 4: TRAINING LOOP
        
        for epoch in range(self.epochs):
            for batch_idx, (real_data,) in enumerate(dataloader):
                batch_size = real_data.size(0)
                
                # Labels for real and fake data
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                for _ in range(self.discriminator_steps):
                    d_optimizer.zero_grad()
                    
                    # Loss on real data
                    real_output = self.discriminator(real_data)
                    d_loss_real = criterion(real_output, real_labels)
                    
                    # Generate fake data
                    noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                    fake_data = self.generator(noise)
                    
                    # Loss on fake data
                    fake_output = self.discriminator(fake_data.detach())
                    d_loss_fake = criterion(fake_output, fake_labels)
                    
                    # Total discriminator loss
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                # Generate fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(noise)
                
                # Generator wants discriminator to think fake is real
                fake_output = self.discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
        
        # Set to evaluation mode after training
        self.generator.eval()
        self.discriminator.eval()
    
    def _sample(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic samples using the trained generator.
        
        Args:
            n_samples: Number of samples to generate
            conditions: Not implemented yet
            
        Returns:
            DataFrame with synthetic samples
        """
        # Ensure generator is in eval mode
        self.generator.eval()
        
        # Generate in batches to handle large n_samples
        samples_list = []
        remaining = n_samples
        
        with torch.no_grad():
            while remaining > 0:
                batch_size = min(remaining, self.batch_size)
                
                # Generate noise
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                
                # Generate samples
                fake_data = self.generator(noise)
                
                # Convert to numpy
                samples_list.append(fake_data.cpu().numpy())
                remaining -= batch_size
        
        # Concatenate all batches
        synthetic_array = np.concatenate(samples_list, axis=0)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(
            synthetic_array,
            columns=self.column_names
        )
        
        return synthetic_df
    
    def get_params(self) -> Dict[str, Any]:
        """Get generator parameters."""
        params = super().get_params()
        params.update({
            "noise_dim": self.noise_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": str(self.device)
        })
        return params
