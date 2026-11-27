"""Loss terms of the beta-VAE. All terms are in nats per individual.

    L = recon + beta |KL(q(z|x) || N(0, I)) - C(t)|

C = 0 gives the plain beta-VAE (Higgins et al., 2017): L = recon + beta KL.
C > 0 gives the capacity form (Burgess et al., 2018): the KL is steered to a capacity that rises
linearly from 0 to C during the first part of training.
"""

import math

import torch
import torch.nn.functional as F

MISSING = 3


def reconstruction_loss(recon_logits, x, mask, dosage=None, likelihood="bernoulli2", focal_gamma=0.0):
    """Negative log-likelihood of the observed calls, per individual.

    bernoulli2: two Bernoullis per SNP on the channels x. categorical / binomial: one distribution
    per SNP on the dosage (0, 1, 2; 3 = missing).

    focal_gamma > 0 (categorical only): focal loss, every call weighted by (1 - p)^gamma, p the
    probability given to the observed genotype, so that rare genotypes count more.
    """
    if likelihood == "bernoulli2":
        bce = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="none")
        return (bce * mask).sum(-1)

    observed = (dosage != MISSING).to(recon_logits.dtype)
    genotype = dosage.clamp(max=2).long()

    if likelihood == "categorical":
        log_p = F.log_softmax(recon_logits.view(recon_logits.shape[0], -1, 3), dim=-1)
        nll = -log_p.gather(-1, genotype.unsqueeze(-1)).squeeze(-1)

        if focal_gamma > 0:
            nll = nll * (1 - torch.exp(-nll)).pow(focal_gamma)
    elif likelihood == "binomial":
        g = genotype.to(recon_logits.dtype)
        nll = -(g * F.logsigmoid(recon_logits) + (2 - g) * F.logsigmoid(-recon_logits))
        nll = nll - math.log(2.0) * (genotype == 1)
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")

    return (nll * observed).sum(-1)


def predicted_genotype(recon_logits, likelihood="bernoulli2"):
    """Most probable genotype per SNP (0, 1, 2); 3 where two Bernoullis give the impossible [0, 0]."""
    if likelihood == "bernoulli2":
        logits = recon_logits.view(recon_logits.shape[0], -1, 2)
        alt, ref = logits[..., 0] > 0, logits[..., 1] > 0
        genotype = torch.full(alt.shape, MISSING, dtype=torch.long, device=recon_logits.device)
        genotype[~alt & ref] = 0
        genotype[alt & ref] = 1
        genotype[alt & ~ref] = 2
        return genotype

    if likelihood == "categorical":
        return recon_logits.view(recon_logits.shape[0], -1, 3).argmax(-1)

    p = torch.sigmoid(recon_logits)
    return torch.stack(((1 - p) ** 2, 2 * p * (1 - p), p ** 2), dim=-1).argmax(-1)


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
