# beta-VAE for Crickets dataset

PyTorch beta-VAE for cricket SNP data: an unsupervised VAE with a Gaussian latent `z`, one model
per LD window.

![beta-VAE architecture](figures/betaVAE_architecture.png)

## Install

From the repository root, create and activate the Conda environment.

```bash
conda env create -n betavae -f environment.yml
conda activate betavae
```

## Run

Set the dataset paths, code paths, and Conda setup in the Slurm scripts for your cluster. Create the log directory specified in `#SBATCH --output` and `#SBATCH --error` before submitting.

### Train all LD blocks

```bash
sbatch run_cricket_all_block.sh
```

This runs five-fold cross-validation for each LD block across scaffolds 1–14, with the same folds as [CAD_VAE_CRICKET](https://github.com/Ziyang-cyber/CAD_VAE_CRICKET).

### Run ablation studies

Select a training job and submit the plotting job after the matching runs finish.

| Experiment | Train | Plot |
| --- | --- | --- |
| KL weight (beta) | `sbatch run_cricket_beta.sh` | `sbatch run_plot_cricket_ablation.sh` |
| Weight and capacity (capacity form) | `sbatch run_cricket_capacity.sh` | `sbatch run_plot_cricket_ablation.sh` |

## Results

![Trait signal per LD window](figures/betaVAE_window_scan.png)
