# Model and training options for cricket SNP windows, all SNPs (24 to 27 Nov 2026)

All results here use every SNP of a window (no LD pruning).

## Answer

- **The earlier model was too narrow.** One hidden layer and 64 to 128 latents, instead of a 512-128 network with
  16 latents, raise the fold-mean venation AUC on test folds:
  - β-VAE: 0.689 to 0.738;
  - JointVAE: 0.692 to 0.731, and 0.742 with γ 1 and one 1024 layer.
- **Almost none of that gain is within population.**
  - Venation AUC within population moves by +0.002 / −0.006 / +0.003 (standard error 0.003 to 0.006).
  - Island accuracy rises by 0.08 to 0.12 at the same time.
  - So the gain is mostly the model reading population structure better (inferred).
- **Shape AUC within population does rise:** by +0.024 / +0.028 (standard error 0.005).
- **Training settings stay:** batch 32, 400 epochs, Adam, learning rate 5e-4, dropout 0.1.
- **Recommended configs:**
  - β-VAE: 128 latents, one 512 layer, β 1.
  - JointVAE: 128 latents, one 1024 layer, γ 1, capacity 80.

## Test folds (read once)

67 windows: the 32 evaluation windows, scaffold 1 blocks 128-147 (venation peak), scaffold 2 blocks 455-474 (shape
peak). Mean over windows of the mean over 3 seeds. Entries are β-VAE / JointVAE.

| Setting | Venation AUC | Venation AUC within population | Shape AUC within population | Island balanced accuracy |
|---|---|---|---|---|
| Earlier model (16 latents, 512-128) | 0.689 / 0.692 | 0.588 / 0.597 | 0.555 / 0.553 | 0.673 / 0.679 |
| 128 latents, one 512 layer | 0.738 / 0.731 | 0.590 / 0.591 | 0.579 / 0.581 | 0.780 / 0.756 |
| JointVAE: 128 latents, one 1024 layer, γ 1 | - / 0.742 | - / 0.594 | - / 0.583 | - / 0.794 |

- 10 principal components of all SNPs: venation 0.680, within population 0.586. No reference with as many
  components as latents was computed.
- Per-window numbers: `test_check_betavae_windows.csv`, `test_check_jointvae_windows.csv` (`base` is the earlier
  model).

## How the settings were chosen (validation crickets only, 36-window panel)

### First screen, one change at a time

β-VAE, one seed. Values are the difference in validation venation AUC from the earlier model; the standard error
of a difference is 0.005 to 0.008.

| Change | Venation AUC |
|---|---|
| One 256 layer | +0.013 |
| 64 latents | +0.011 |
| Categorical or binomial decoder, 0 / 0.5 / 1 input, missing-mask channel, input masking, weight decay, PCA input | within ±0.005 |
| Focal loss, convolution or block encoder, dropout 0.3, 128-64 network | −0.011 to −0.027 |
| Block decoder, best-validation epoch | −0.031 |
| 4 latents | −0.057 |

### Latents and layer shape

The two changes act together: 64 latents with one 256 layer gives +0.044 (3 seeds), against +0.011 and +0.013 apart.

Shape grid, β-VAE, one seed, validation venation AUC:

| Latents | 512-128 | 512-512 | One 256 | One 512 | One 1024 |
|---|---|---|---|---|---|
| 16 | 0.666 | 0.684 | 0.681 | 0.685 | 0.683 |
| 32 | 0.677 | 0.687 | 0.701 | 0.705 | 0.700 |
| 64 | 0.676 | 0.691 | 0.713 | 0.715 | 0.718 |
| 128 | 0.678 | 0.691 | 0.709 | 0.717 | 0.722 |
| 256 | 0.676 | 0.683 | 0.708 | 0.717 | 0.721 |

- **The second hidden layer is the bottleneck.** With it the score stays at 0.68 and island accuracy at 0.71.
- With one hidden layer, the score rises up to 64 latents and is flat from 64 to 256.
- JointVAE follows the same pattern: 0.670 to 0.677 with 512-128, and 0.714 to 0.717 with one 1024 layer at 64 or
  more latents.

### Training settings

Differences from 128 latents with one 512 layer, one seed, β-VAE / JointVAE:

| Change | Venation AUC |
|---|---|
| Batch 16 | −0.001 / not run |
| Batch 64 | −0.006 / −0.007 |
| Batch 64, learning rate 1e-3 | −0.003 / not run |
| Batch 128 | −0.010 / −0.008 |
| Batch 128, learning rate 2e-3 | −0.008 / not run |
| Learning rate 1e-4 | −0.015 / −0.021 |
| Learning rate 2e-4 | −0.002 / −0.007 |
| Learning rate 1e-3 | −0.003 / +0.003 |
| 200 epochs | −0.009 / −0.012 |
| 800 epochs | +0.003 / +0.006 |
| Learning rate 1e-4, 800 epochs | −0.002 / not run |
| zzhang's recipe (learning rate 1e-4, 200 epochs, AdamW, weight decay 1e-5) | −0.030 / not run |
| Dropout 0 | +0.001 / +0.002 |
| Dropout 0.2 | +0.001 / not run |
| β warm-up over 100 epochs | +0.001 / not applicable |

- **Too few gradient steps is what hurts.** A low learning rate, few epochs and zzhang's recipe all lose.
- **Batch 32 is the same as zzhang's CAD-VAE.** Larger batches train 2 to 3 times faster but lose up to 0.010.
- **400 epochs is close to enough:** 800 epochs adds little and doubles the time.

### Loss weights and confirmation, 3 seeds

Validation venation AUC. Differences are paired over windows, against 128 latents with one 512 layer.

| Config | Venation AUC | Difference | Island accuracy |
|---|---|---|---|
| β-VAE earlier model (16 latents, 512-128) | 0.663 | −0.057 ± 0.007 | 0.717 |
| β-VAE 128 latents, one 512 layer, β 1 | 0.720 | | 0.822 |
| β-VAE one 1024 layer | 0.722 | +0.002 ± 0.003 | 0.826 |
| β-VAE 64 latents, one 1024 layer | 0.719 | −0.001 ± 0.003 | 0.829 |
| β-VAE β 0.5 | 0.720 | 0.000 ± 0.002 | 0.829 |
| β-VAE 800 epochs | 0.721 | +0.001 ± 0.002 | 0.827 |
| β-VAE one 1024 layer, β 0.5 | 0.723 | +0.002 ± 0.003 | 0.833 |
| JointVAE earlier model (16 latents, 512-128, capacity 40) | 0.667 | −0.042 ± 0.005 | 0.715 |
| JointVAE 128 latents, one 512 layer, γ 10, capacity 80 | 0.709 | | 0.800 |
| JointVAE γ 1 | 0.719 | +0.010 ± 0.003 | 0.833 |
| JointVAE capacity 160 | 0.713 | +0.004 ± 0.001 | 0.809 |
| JointVAE one 1024 layer | 0.716 | +0.007 ± 0.003 | 0.819 |
| JointVAE one 1024 layer, γ 1 | 0.724 | +0.015 ± 0.003 | 0.831 |
| JointVAE γ 0.1 (one seed) | 0.727 | +0.018 ± 0.005 | 0.837 |

- **β-VAE is at a plateau.** β 2 and 4 lose 0.004 and 0.010 (one seed).
- **JointVAE gains from a weaker latent penalty (γ 1).** It then matches the β-VAE.
- Seeds agree within 0.006 (median spread of per-seed means).

## Measured and inferred

- **Measured:** every number above. Selection used validation crickets only; test folds were read for the final
  table.
- **Inferred:** that the fold-mean gain is population structure. It rests on two measurements: the flat
  within-population AUC and the rise in island and population accuracy.
- **Not tested:**
  - conditioning on population (set aside because it is no longer unsupervised);
  - a β-VAE test-fold check of the one 1024 layer config, because its validation difference is too small.

## Files

- Settings (one line per run, "name<TAB>extra arguments of train.py"):
  - `settings_screen.txt`, `settings_screen2.txt`: first screen.
  - `settings_levels.txt`, `settings_seeds.txt`: earlier model and model shape, 3 seeds.
  - `settings_unpruned.txt`, `settings_unpruned2.txt`: shape grid, training settings, confirmation.
  - `settings_final.txt`, `settings_final3.txt`: test-fold runs.
- Windows: `windows_panel36.txt` (validation panel), `windows_final.txt` (test windows).
- `summary.py`: tables per setting. `test_check_*_windows.csv`: per-window test numbers.
- Results on gruffalo: `/mnt/shared/scratch/weiyeli/crop_vae/results_cricket/mirror/{betavae,jointvae}/explore*`
  (scratch is purged automatically).
