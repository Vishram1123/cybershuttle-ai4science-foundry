#!/bin/bash
#SBATCH --job-name=hyenadna-inference
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hyenadna-inference-%j.out
#
# Reproducible HyenaDNA inference job for PACE ICE (ice-gpu partition).
# Builds the environment (idempotent — skips install if already present),
# runs the smoke test + validation check, then embeds the requested input.
#
# Usage: sbatch hyenadna_inference.sbatch [path/to/input.fasta]

set -euo pipefail
mkdir -p logs

REPO_DIR="$HOME/scratch/hyena-dna"
ENV_PREFIX="$HOME/scratch/.conda/envs/hyena-dna"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
    bash "$SCRIPT_DIR/hyenadna_setup_env.sh" "$ENV_PREFIX" "$REPO_DIR"
fi
conda activate "$ENV_PREFIX"

export HYENADNA_REPO="$REPO_DIR"
export HF_HOME="$HOME/scratch/.cache/huggingface"  # must match setup script, or the checkpoint re-downloads

python "$SCRIPT_DIR/hyenadna_validate.py" --model hyenadna-tiny-1k-seqlen

if [ -n "${1:-}" ]; then
    python "$SCRIPT_DIR/hyenadna_inference.py" \
        --model hyenadna-tiny-1k-seqlen \
        --fasta "$1" \
        --output "results/hyenadna-embeddings-${SLURM_JOB_ID}.pt"
fi
