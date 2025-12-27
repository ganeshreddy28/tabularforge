"""
TVAE Generator (Tabular Variational Autoencoder)
=================================================

This module implements TVAE (Tabular Variational Autoencoder) for synthetic
data generation. VAEs learn a compressed latent representation of the data
and can generate new samples by decoding random points in this latent space.

How VAEs Work:
    1. Encoder: Maps input data → latent distribution (mean, variance)
    2. Sampling: Sample from the latent distribution using reparameterization
    3. Decoder: Maps latent sample → reconstructed data
    
Key Innovation in TVAE:
    - Mode-specific normalization for continuous columns (same as CTGAN)
    - Handles mixed data types (continuous + categorical)
    - Uses KL divergence to ensure latent space is well-structured

Advantages over GANs:
    - More stable training (no adversarial dynamics)
    - Explicit latent space for interpolation
    - Can measure reconstruction quality
    
Author: Sai Ganesh Kolan
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tabularforge.generators.base import BaseGenerator


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class Encoder(nn.Module):
    """
    Encoder network for TVAE.
    
    The encoder maps input data to a latent distribution, outputting the
    mean and log-variance of a Gaussian distribution in latent space.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int = 2
    ) -> None:
        super().__init__()
        
        # Build encoder layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.encoder_body = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mean(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Decoder network for TVAE."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2
    ) -> None:
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VAE(nn.Module):
    """Complete VAE model combining Encoder and Decoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_layers: int = 2
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_layers)
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var
    
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim).to(device)
        return self.decoder(z)


class TVAEGenerator(BaseGenerator):
    """
    TVAE-based synthetic data generator.
    
    TVAE (Tabular Variational Autoencoder) learns a compressed latent
    representation of tabular data and can generate new samples by
    decoding random points from this latent space.
    
    Example:
        >>> generator = TVAEGenerator(epochs=300)
        >>> generator.fit(data, encoder)
        >>> synthetic = generator.sample(1000)
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        epochs: int = 300,
        batch_size: int = 500,
        learning_rate: float = 1e-3,
        random_state: Optional[int] = None
    ) -> None:
        super().__init__(random_state=random_state)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.vae: Optional[VAE] = None
        self._data_dim: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _vae_loss(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + kl_loss
    
    def _fit(self, data: pd.DataFrame, encoder: Any) -> None:
        data_array = data.values.astype(np.float32)
        self._data_dim = data_array.shape[1]
        self.column_names = list(data.columns)
        
        data_tensor = torch.tensor(data_array, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.vae = VAE(
            input_dim=self._data_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers
        ).to(self.device)
        
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        self.vae.train()
        for epoch in range(self.epochs):
            for batch_idx, (batch_data,) in enumerate(dataloader):
                optimizer.zero_grad()
                reconstruction, mean, log_var = self.vae(batch_data)
                loss = self._vae_loss(reconstruction, batch_data, mean, log_var)
                loss.backward()
                optimizer.step()
        
        self.vae.eval()
    
    def _sample(self, n_samples: int, conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        self.vae.eval()
        with torch.no_grad():
            samples = self.vae.sample(n_samples, self.device)
            synthetic_array = samples.cpu().numpy()
        
        return pd.DataFrame(synthetic_array, columns=self.column_names)
    
    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "device": str(self.device)
        })
        return params
