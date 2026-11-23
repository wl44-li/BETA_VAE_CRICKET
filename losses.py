"""Loss terms of the beta-VAE. All terms are in nats per individual.

    L = recon + beta |KL(q(z|x) || N(0, I)) - C(t)|

C = 0 gives the plain beta-VAE (Higgins et al., 2017): L = recon + beta KL.
C > 0 gives the capacity form (Burgess et al., 2018): the KL is steered to a capacity that rises
linearly from 0 to C during the first part of training.
"""

import torch.nn.functional as F


def reconstruction_loss(recon_logits, x, mask):
    """Bernoulli negative log-likelihood of the observed entries, per individual."""
    bce = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="none")
    return (bce * mask).sum(-1)


def kl_gaussian(mu, logvar):
    """KL(q(z|x) || N(0, I)) per individual and latent dimension -> [B, z_dim]."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def capacity(step, ramp_steps, c_max):
    """Channel capacity at a training step: linear from 0 to c_max over ramp_steps."""
    if ramp_steps <= 0:
        return c_max

    return c_max * min(1.0, step / ramp_steps)


def beta_vae_objective(loss_recon, loss_kl, beta, capacity_z):
    return loss_recon + beta * (loss_kl - capacity_z).abs()
