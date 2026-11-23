"""beta-VAE with an MLP backbone for cricket SNP windows.

One Gaussian latent z. The backbone has the widths of the CAD-VAE and cVAE models for the same
data (512-128, 16 latents).

Input: every SNP is two binary channels (ALT presence, REF presence), as in the stored zarr files.
A missing call is [0, 0] there; here it is masked: replaced by the training mean in the encoder
input and left out of the reconstruction loss.
"""

from dataclasses import dataclass

import torch
from torch import nn

MISSING = 3


@dataclass
class ModelConfig:
    z_dim: int = 16

    hidden_dims: tuple = (512, 128)
    dropout: float = 0.1


def encode_genotype(dosage):
    """uint8 dosage [B, n] (3 = missing) -> channels [B, 2n] and the mask of observed entries."""
    x = torch.stack(((dosage == 1) | (dosage == 2), dosage < 2), dim=-1)
    x = x.to(torch.float32).flatten(1)
    mask = (dosage != MISSING).repeat_interleave(2, dim=1).to(torch.float32)
    return x, mask


def mlp(input_dim, hidden_dims, dropout):
    """Build MLP hidden layers without a final output layer."""
    layers = []

    for width in hidden_dims:
        layers += [nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU()]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        input_dim = width

    return nn.Sequential(*layers)


class BetaVAE(nn.Module):
    """q(z | x) Gaussian; p(x | z) Bernoulli; p(z) = N(0, I)."""

    def __init__(self, input_dim, cfg=None):
        super().__init__()
        cfg = cfg or ModelConfig()

        self.cfg = cfg
        self.input_dim = input_dim
        self.z_dim = cfg.z_dim

        self.encoder = mlp(input_dim, cfg.hidden_dims, cfg.dropout)

        hidden = cfg.hidden_dims[-1]
        self.fc_mu = nn.Linear(hidden, cfg.z_dim)
        self.fc_logvar = nn.Linear(hidden, cfg.z_dim)

        self.decoder = nn.Sequential(
            mlp(cfg.z_dim, tuple(reversed(cfg.hidden_dims)), cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], input_dim),
        )

        # Training mean of every input channel (the allele-frequency information).
        self.register_buffer("input_mean", torch.zeros(input_dim))

    @torch.no_grad()
    def set_data_stats(self, feature_mean):
        """Centre the encoder input and start the decoder at the allele frequencies."""
        self.input_mean.copy_(feature_mean)

        p = feature_mean.clamp(1e-3, 1 - 1e-3)
        self.decoder[-1].bias.copy_(torch.log(p) - torch.log1p(-p))

    def encode(self, x, mask):
        """Return (mu, logvar); missing entries enter as the training mean."""
        hidden = self.encoder(torch.where(mask > 0, x, self.input_mean) - self.input_mean)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, mask, sample=True):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar
