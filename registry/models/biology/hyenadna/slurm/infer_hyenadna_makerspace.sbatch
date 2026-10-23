#!/bin/bash
# HyenaDNA pretrained inference on GT PACE ICE / AI Makerspace (ice-gpu partition).
#
# Submit from registry/models/biology/hyenadna/ with:
#   sbatch slurm/infer_hyenadna_makerspace.sbatch
#
# This partition's --gres=gpu:1 can allocate a V100, A100, H100, or H200 node
# (AI Makerspace = the H100/H200 portion of ice-gpu) — or even a non-CUDA AMD
# MI210 node. This job requests an H100 specifically (--gres=gpu:h100:1) to
# exercise the AI Makerspace hardware the semester plan is targeting;
# environment/setup_env.sh detects whichever GPU you actually got and installs
# a matching PyTorch/CUDA build either way, so this script works unmodified
# if you swap the gres type to target a different generation.

#SBATCH --job-name=hyenadna-infer
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hyenadna-infer-%j.out
#SBATCH --error=logs/hyenadna-infer-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

echo "== Job ${SLURM_JOB_ID} on $(hostname) =="
echo "Allocated GPU(s):"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || echo "nvidia-smi not available"
echo

bash environment/setup_env.sh

# see environment/setup_env.sh for why -u is relaxed here: conda's own
# activate hooks on this cluster aren't `set -u`-safe.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hyenadna-infer
set -u

echo
echo "== Running inference =="
python scripts/run_inference.py \
  --checkpoint hyenadna-tiny-1k-seqlen \
  --input_fasta data/NM_000518.5_HBB_mRNA.fasta \
  --output_dir results \
  --device cuda

echo
echo "== Validating output =="
python scripts/validate_output.py \
  --results_dir results \
  --report results/validation_report.json

echo
echo "== Done. See results/ for outputs and logs/hyenadna-infer-${SLURM_JOB_ID}.out for the full log. =="
