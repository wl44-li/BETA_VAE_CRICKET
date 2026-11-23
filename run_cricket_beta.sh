#!/bin/bash
#SBATCH --job-name=betavae_cricket_beta
#SBATCH --output=/mnt/shared/scratch/weiyeli/crop_vae/logs/betavae_cricket_beta_%A_%a.out
#SBATCH --error=/mnt/shared/scratch/weiyeli/crop_vae/logs/betavae_cricket_beta_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

# Plain beta-VAE: weight of the KL term.
# Runs on the 12 tuning windows; see ablation_common.sh.

set -euo pipefail

BETA_VALUES=(0.25 1 4 16 64 100 256)
CAPACITY_VALUES=(0)
DROPOUT_VALUES=(0.1 0.5)

source /mnt/shared/scratch/weiyeli/crop_vae/repos/BETA_VAE_CRICKET/ablation_common.sh
