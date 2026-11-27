"""beta-VAE with an MLP backbone for cricket SNP windows.

One Gaussian latent z. The default backbone has the widths of the CAD-VAE and cVAE models for the
same data (512-128, 16 latents).

Input: every SNP is two binary channels (ALT presence, REF presence), as in the stored zarr files.
A missing call is [0, 0] there; here it is masked: replaced by the training mean in the encoder
input and left out of the reconstruction loss.

Options (the defaults are the model of the all-blocks run):
    likelihood  bernoulli2   two independent Bernoullis per SNP (ALT presence, REF presence)
                categorical  one 3-class categorical per SNP (genotype 0 / 1 / 2)
                binomial     Binomial(2, p) per SNP
    encoder     mlp          fully connected on all inputs
                pca          fixed projection on the leading principal components of the training
                             crickets (input_pca of them), then the MLP
                block        one small linear map per block of consecutive SNPs, then the MLP
                conv         three strided 1-D convolutions along the SNP order, then the MLP
    decoder     mlp          fully connected output layer
                block        one small linear map per block of consecutive SNPs
    input_encoding  channels  the two channels per SNP
                    dosage    one value per SNP, 0 / 0.5 / 1 (popVAE, GenoCAE); MLP encoder only
    mask_channel    adds one input per SNP that says whether the call is missing (GenoCAE); MLP
                    encoder only
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

MISSING = 3

OUTPUTS_PER_SNP = {"bernoulli2": 2, "categorical": 3, "binomial": 1}


@dataclass
class ModelConfig:
    z_dim: int = 16

    hidden_dims: tuple = (512, 128)
    dropout: float = 0.1

    likelihood: str = "bernoulli2"
    encoder: str = "mlp"
    decoder: str = "mlp"

    input_pca: int = 50   # components, for encoder = "pca"
    block_size: int = 50  # SNPs per block, for the block encoder / decoder
    block_dim: int = 4    # units per block

    input_encoding: str = "channels"
    mask_channel: bool = False


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


class BlockLinear(nn.Module):
    """An own linear map for every block: [B, n_blocks, d_in] -> [B, n_blocks, d_out]."""

    def __init__(self, n_blocks, d_in, d_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_blocks, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_blocks, d_out)) if bias else None
        nn.init.uniform_(self.weight, -1 / math.sqrt(d_in), 1 / math.sqrt(d_in))

    def forward(self, x):
        out = torch.einsum("bnk,nkd->bnd", x, self.weight)
        return out if self.bias is None else out + self.bias


class BlockFrontEnd(nn.Module):
    """Consecutive blocks of SNPs, each with its own small linear map."""

    def __init__(self, n_snps, block_size, block_dim):
        super().__init__()
        self.block_size = block_size
        self.n_blocks = math.ceil(n_snps / block_size)
        self.pad = 2 * (self.n_blocks * block_size - n_snps)
        self.linear = BlockLinear(self.n_blocks, 2 * block_size, block_dim)
        self.out_dim = self.n_blocks * block_dim

    def forward(self, x):
        x = F.pad(x, (0, self.pad)).view(x.shape[0], self.n_blocks, 2 * self.block_size)
        return F.gelu(self.linear(x)).flatten(1)


class ConvFrontEnd(nn.Module):
    """Three strided convolutions along the SNP order."""

    def __init__(self, n_snps):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=9, stride=4, padding=4), nn.GELU(),
        )

        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, 2, n_snps)).flatten(1).shape[1]

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1, 2).transpose(1, 2)).flatten(1)


class BlockHead(nn.Module):
    """Decoder output layer with one small linear map per block of consecutive SNPs."""

    def __init__(self, hidden, n_snps, per_snp, block_size, block_dim):
        super().__init__()
        self.n_blocks = math.ceil(n_snps / block_size)
        self.block_dim = block_dim
        self.out_dim = n_snps * per_snp
        self.expand = nn.Linear(hidden, self.n_blocks * block_dim)
        self.linear = BlockLinear(self.n_blocks, block_dim, block_size * per_snp, bias=False)

    def forward(self, hidden):
        blocks = F.gelu(self.expand(hidden)).view(hidden.shape[0], self.n_blocks, self.block_dim)
        return self.linear(blocks).flatten(1)[:, :self.out_dim]


class BetaVAE(nn.Module):
    """q(z | x) Gaussian; p(x | z) per SNP (see the module text); p(z) = N(0, I)."""

    def __init__(self, input_dim, cfg=None):
        super().__init__()
        cfg = cfg or ModelConfig()

        self.cfg = cfg
        self.input_dim = input_dim
        self.n_snps = input_dim // 2
        self.z_dim = cfg.z_dim
        self.output_dim = self.n_snps * OUTPUTS_PER_SNP[cfg.likelihood]

        # Encoder: front end (what the first layer sees) and the MLP.
        if cfg.encoder != "mlp" and (cfg.input_encoding != "channels" or cfg.mask_channel):
            raise ValueError("input_encoding and mask_channel need the MLP encoder.")

        if cfg.encoder == "mlp":
            per_snp = (2 if cfg.input_encoding == "channels" else 1) + int(cfg.mask_channel)
            self.front_end, front_dim = None, self.n_snps * per_snp
        elif cfg.encoder == "pca":
            self.front_end, front_dim = None, cfg.input_pca
            self.register_buffer("pca_projection", torch.zeros(input_dim, cfg.input_pca))
        elif cfg.encoder == "block":
            self.front_end = BlockFrontEnd(self.n_snps, cfg.block_size, cfg.block_dim)
            front_dim = self.front_end.out_dim
        elif cfg.encoder == "conv":
            self.front_end = ConvFrontEnd(self.n_snps)
            front_dim = self.front_end.out_dim
        else:
            raise ValueError(f"Unknown encoder: {cfg.encoder}")

        self.encoder = mlp(front_dim, cfg.hidden_dims, cfg.dropout)

        hidden = cfg.hidden_dims[-1]
        self.fc_mu = nn.Linear(hidden, cfg.z_dim)
        self.fc_logvar = nn.Linear(hidden, cfg.z_dim)

        # Decoder: the MLP, an output layer without bias, and the output bias (set to the
        # frequencies of the training crickets).
        self.decoder_body = mlp(cfg.z_dim, tuple(reversed(cfg.hidden_dims)), cfg.dropout)

        if cfg.decoder == "mlp":
            self.decoder_head = nn.Linear(cfg.hidden_dims[0], self.output_dim, bias=False)
        elif cfg.decoder == "block":
            self.decoder_head = BlockHead(
                cfg.hidden_dims[0], self.n_snps, OUTPUTS_PER_SNP[cfg.likelihood],
                cfg.block_size, cfg.block_dim,
            )
        else:
            raise ValueError(f"Unknown decoder: {cfg.decoder}")

        self.output_bias = nn.Parameter(torch.zeros(self.output_dim))

        # Training mean of every input channel (the allele-frequency information).
        self.register_buffer("input_mean", torch.zeros(input_dim))

    @torch.no_grad()
    def set_data_stats(self, feature_mean, genotype_frequency, allele_frequency):
        """Centre the encoder input and start the decoder at the training frequencies.

        feature_mean [2n]: mean of the two channels; genotype_frequency [n, 3]; allele_frequency [n].
        """
        self.input_mean.copy_(feature_mean)

        if self.cfg.likelihood == "bernoulli2":
            p = feature_mean.clamp(1e-3, 1 - 1e-3)
            bias = torch.log(p) - torch.log1p(-p)
        elif self.cfg.likelihood == "categorical":
            bias = torch.log(genotype_frequency.clamp_min(1e-3)).flatten()
        else:
            p = allele_frequency.clamp(1e-3, 1 - 1e-3)
            bias = torch.log(p) - torch.log1p(-p)

        self.output_bias.copy_(bias)

    @torch.no_grad()
    def fit_input_pca(self, x, mask):
        """Leading principal components of the centred training inputs (encoder = "pca")."""
        centred = torch.where(mask > 0, x, self.input_mean) - self.input_mean
        values, vectors = torch.linalg.eigh(centred @ centred.T)

        order = torch.argsort(values, descending=True)[:self.cfg.input_pca]
        singular = values[order].clamp_min(1e-8).sqrt()

        # Unit-length directions, scaled so that the first component has unit variance.
        directions = centred.T @ vectors[:, order] / singular
        self.pca_projection.copy_(directions * math.sqrt(x.shape[0]) / singular[0])

    def encode(self, x, mask):
        """Return (mu, logvar); missing entries enter as the training mean."""
        centred = torch.where(mask > 0, x, self.input_mean) - self.input_mean

        if self.cfg.input_encoding == "dosage":
            # dosage / 2 = (ALT presence + 1 - REF presence) / 2, centred on its training mean.
            centred = (centred[:, 0::2] - centred[:, 1::2]) / 2

        if self.cfg.mask_channel:
            centred = torch.cat((centred, mask[:, 0::2] - 1.0), dim=1)  # 0 observed, -1 missing

        if self.cfg.encoder == "pca":
            centred = centred @ self.pca_projection
        elif self.front_end is not None:
            centred = self.front_end(centred)

        hidden = self.encoder(centred)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder_head(self.decoder_body(z)) + self.output_bias

    def forward(self, x, mask, sample=True):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar
