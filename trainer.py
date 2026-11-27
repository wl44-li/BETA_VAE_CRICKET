"""Train a beta-VAE on the genotypes of one fold. No label enters this file."""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from losses import (
    MISSING,
    beta_vae_objective,
    capacity,
    kl_gaussian,
    predicted_genotype,
    reconstruction_loss,
)
from model import encode_genotype


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 400
    lr: float = 5e-4
    weight_decay: float = 0.0

    # capacity_z = 0: plain beta-VAE, loss = recon + beta KL.
    # capacity_z > 0: loss = recon + beta |KL - C(t)|, C rising from 0 to capacity_z over the
    # first ramp_fraction of the training steps (use a large beta, e.g. 1000).
    beta: float = 16.0
    capacity_z: float = 0.0
    ramp_fraction: float = 0.5
    beta_warmup_epochs: int = 0  # linear warm-up of beta, as in the cVAE / CAD-VAE trainers

    # "last": the model after the last epoch.
    # "best_val": lowest validation loss among the epochs after the warm-up and the ramp.
    select: str = "last"

    # Share of the observed SNPs hidden from the encoder at every training step (the loss still
    # covers them). 0: off.
    input_mask_rate: float = 0.0

    # Focal loss exponent for the categorical likelihood, training loss only. 0: off.
    focal_gamma: float = 0.0

    verbose: bool = True
    log_every: int = 50


def get_beta(epoch, beta, warmup_epochs):
    """Return the KL coefficient for a one-based epoch index."""
    if warmup_epochs <= 0:
        return float(beta)

    return float(beta) * min(1.0, epoch / warmup_epochs)


class BetaVAETrainer:
    """Train a beta-VAE on uint8 dosage arrays (3 = missing)."""

    def __init__(self, model, cfg: TrainConfig, device=None):
        self.cfg = cfg

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)

        optimizer_class = AdamW if cfg.weight_decay > 0 else Adam
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=self.device.type == "cuda",
        )

        self.history = []
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_state_dict = None
        self.reference = None
        self.common_genotype = None
        self.likelihood = model.cfg.likelihood

    def _tensor(self, dosage):
        return torch.as_tensor(np.asarray(dosage), dtype=torch.uint8, device=self.device)

    def _forward_losses(self, dosage, sample, beta, capacity_z, input_mask_rate=0.0, focal_gamma=0.0):
        x, mask = encode_genotype(dosage)
        encoder_mask = mask

        if input_mask_rate > 0:
            hidden = torch.rand(dosage.shape, device=dosage.device) < input_mask_rate
            encoder_mask = mask * (~hidden).repeat_interleave(2, dim=1)

        recon_logits, mu, logvar = self.model(x, encoder_mask, sample=sample)

        loss_recon = reconstruction_loss(
            recon_logits, x, mask, dosage, self.likelihood, focal_gamma).mean()
        loss_kl = kl_gaussian(mu, logvar).sum(-1).mean()

        loss = beta_vae_objective(loss_recon, loss_kl, beta, capacity_z)

        metrics = {
            "loss": loss.detach(),
            "recon": loss_recon.detach(),
            "kl": loss_kl.detach(),
        }

        return loss, metrics

    @torch.no_grad()
    def _final_objective(self, dosage):
        """Objective with the final beta and capacity, z = mean."""
        self.model.eval()

        _, metrics = self._forward_losses(
            dosage,
            sample=False,
            beta=self.cfg.beta,
            capacity_z=self.cfg.capacity_z,
        )

        return {name: value.item() for name, value in metrics.items()}

    def fit(self, x_train, x_val=None):
        """Train on x_train. x_val is only used to follow (and optionally select by) its loss."""
        cfg = self.cfg

        if cfg.focal_gamma > 0 and self.likelihood != "categorical":
            raise ValueError("focal_gamma needs the categorical likelihood.")

        if cfg.select not in ("last", "best_val"):
            raise ValueError("select must be 'last' or 'best_val'.")

        train = self._tensor(x_train)
        val = self._tensor(x_val) if x_val is not None and len(x_val) > 0 else None

        if cfg.select == "best_val" and val is None:
            raise ValueError("select='best_val' needs a validation set.")

        # Allele-frequency information of the training crickets only.
        x, mask = encode_genotype(train)
        feature_mean = x.sum(0) / mask.sum(0).clamp_min(1)

        counts = torch.stack([(train == g).sum(0) for g in range(3)], dim=-1).to(torch.float32)
        genotype_frequency = (counts + 0.5) / (counts.sum(-1, keepdim=True) + 1.5)
        allele_frequency = (counts[:, 1] + 2 * counts[:, 2] + 0.5) / (2 * counts.sum(-1) + 1.0)

        self.model.set_data_stats(feature_mean, genotype_frequency, allele_frequency)

        if self.model.cfg.encoder == "pca":
            self.model.fit_input_pca(x, mask)

        del x, mask

        self.reference = feature_mean.clamp(1e-4, 1 - 1e-4)
        self.common_genotype = counts.argmax(-1)

        n_train = train.shape[0]
        steps_per_epoch = math.ceil(n_train / cfg.batch_size)
        ramp_steps = (
            int(cfg.ramp_fraction * cfg.max_epochs * steps_per_epoch)
            if cfg.capacity_z > 0
            else 0
        )
        schedule_end = max(ramp_steps, cfg.beta_warmup_epochs * steps_per_epoch)

        self.history = []
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_state_dict = None

        step = 0

        for epoch in range(1, cfg.max_epochs + 1):
            self.model.train()

            beta = get_beta(epoch, cfg.beta, cfg.beta_warmup_epochs)
            order = torch.randperm(n_train, device=self.device)
            totals = {}

            for batch_id in range(steps_per_epoch):
                batch = order[batch_id * cfg.batch_size:(batch_id + 1) * cfg.batch_size]

                loss, metrics = self._forward_losses(
                    train[batch],
                    sample=True,
                    beta=beta,
                    capacity_z=capacity(step, ramp_steps, cfg.capacity_z),
                    input_mask_rate=cfg.input_mask_rate,
                    focal_gamma=cfg.focal_gamma,
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                step += 1

                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value

            # One transfer from the GPU per epoch, not per step.
            totals = {name: value.item() for name, value in totals.items()}

            if not math.isfinite(totals["loss"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")

            row = {
                "epoch": epoch,
                "beta": beta,
                "capacity_z": capacity(step, ramp_steps, cfg.capacity_z),
            }

            row.update({
                f"train_{name}": value / steps_per_epoch
                for name, value in totals.items()
            })

            if val is not None:
                val_metrics = self._final_objective(val)

                row.update({
                    f"val_{name}": value
                    for name, value in val_metrics.items()
                })

                if (
                    cfg.select == "best_val"
                    and step >= schedule_end
                    and val_metrics["loss"] < self.best_val_loss
                ):
                    self.best_val_loss = val_metrics["loss"]
                    self.best_epoch = epoch

                    self.best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }

            self.history.append(row)

            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == cfg.max_epochs):
                print(
                    f"Epoch {epoch:03d}/{cfg.max_epochs} "
                    f"beta={beta:.4g} "
                    f"train_loss={row['train_loss']:.2f} "
                    f"train_recon={row['train_recon']:.2f} "
                    f"kl={row['train_kl']:.3f} "
                    f"val_loss={row.get('val_loss', float('nan')):.2f}"
                )

        if cfg.select == "last":
            self.best_epoch = cfg.max_epochs
            self.best_val_loss = self.history[-1].get("val_loss", float("nan"))

        return self.history

    def restore_best_model(self):
        """Load the selected weights (nothing to do when the last epoch is used)."""
        if self.cfg.select == "last":
            return

        if self.best_state_dict is None:
            raise RuntimeError("No best model is available. Run fit() first.")

        self.model.load_state_dict(self.best_state_dict)

    @torch.no_grad()
    def evaluate(self, x, batch_size=128):
        """Label-free numbers of one split, with z = mean."""
        dosage = self._tensor(x)

        if dosage.shape[0] == 0:
            return {}

        self.model.eval()

        recon, n_observed, reference, kl = [], [], [], []
        correct, correct_common, n_calls = 0.0, 0.0, 0.0

        for start in range(0, dosage.shape[0], batch_size):
            batch = dosage[start:start + batch_size]
            x_batch, mask = encode_genotype(batch)
            recon_logits, mu, logvar = self.model(x_batch, mask, sample=False)

            recon.append(reconstruction_loss(recon_logits, x_batch, mask, batch, self.likelihood))
            n_observed.append(mask.sum(-1))

            observed = batch != MISSING
            correct += ((predicted_genotype(recon_logits, self.likelihood) == batch) & observed).sum().item()
            correct_common += ((self.common_genotype == batch) & observed).sum().item()
            n_calls += observed.sum().item()
            reference.append(
                (
                    F.binary_cross_entropy(
                        self.reference.expand_as(x_batch), x_batch, reduction="none"
                    ) * mask
                ).sum(-1)
            )
            kl.append(kl_gaussian(mu, logvar))

        recon = torch.cat(recon)
        n_observed = torch.cat(n_observed).clamp_min(1)
        reference = torch.cat(reference)
        kl = torch.cat(kl)

        metrics = self._final_objective(dosage)

        metrics.update({
            # Reconstruction per observed input entry, and the same for a model that
            # only knows the training allele frequencies.
            "recon_per_feature": (recon / n_observed).mean().item(),
            "allele_frequency_recon_per_feature": (reference / n_observed).mean().item(),
            # Negative log-likelihood per observed SNP (comparable within one likelihood only)
            # and the share of calls reconstructed exactly (comparable across likelihoods).
            "recon_per_snp": (2 * recon / n_observed).mean().item(),
            "concordance": correct / max(n_calls, 1.0),
            "common_genotype_concordance": correct_common / max(n_calls, 1.0),
            "n_active_z": int((kl.mean(0) > 0.01).sum().item()),
        })

        return metrics

    @torch.no_grad()
    def encode_dataset(self, x, batch_size=128):
        """Posterior mean and log-variance of z, in the stored sample order."""
        dosage = self._tensor(x)

        was_training = self.model.training
        self.model.eval()

        outputs = {"mu": [], "logvar": []}

        try:
            for start in range(0, dosage.shape[0], batch_size):
                x_batch, mask = encode_genotype(dosage[start:start + batch_size])
                mu, logvar = self.model.encode(x_batch, mask)

                outputs["mu"].append(mu.cpu().numpy())
                outputs["logvar"].append(logvar.cpu().numpy())
        finally:
            self.model.train(was_training)

        return {
            name: (
                np.concatenate(values, axis=0)
                if values
                else np.zeros((0, self.model.z_dim), dtype=np.float32)
            )
            for name, values in outputs.items()
        }
