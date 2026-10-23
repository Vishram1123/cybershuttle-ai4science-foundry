#!/bin/bash
# Set up a HyenaDNA conda environment, picking a PyTorch build that matches
# the GPU actually allocated to this job. Run on a GPU node (compute
# capability detection needs nvidia-smi to see a real GPU) — on PACE ICE
# that means inside an `srun`/`sbatch` allocation on the ice-gpu partition,
# not on the login node.
#
# Usage: ./hyenadna_setup_env.sh [env_prefix] [repo_dir]
set -euo pipefail

ENV_PREFIX="${1:-$HOME/scratch/.conda/envs/hyena-dna}"
REPO_DIR="${2:-$HOME/scratch/hyena-dna}"

# PACE ICE home quota (~30GB) is consumed fast by package caches long before
# scratch fills up. Every cache that defaults under $HOME needs redirecting.
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export TMPDIR="$HOME/scratch/tmp"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export GIT_CONFIG_GLOBAL="$HOME/scratch/.gitconfig"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"

if [ ! -d "$REPO_DIR" ]; then
    git clone --recurse-submodules https://github.com/HazyResearch/hyena-dna.git "$REPO_DIR"
fi
cd "$REPO_DIR"

module load miniforge
mamba create -y -p "$ENV_PREFIX" python=3.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Pick a PyTorch build that actually has compiled kernels for the allocated
# GPU. The repo's documented pin, torch==1.13.0+cu117, only ships kernels
# through compute capability 8.6 (Ampere); it silently no-ops ("no kernel
# image is available") on Hopper (compute capability 9.0: H100/H200), which
# PACE ICE's ice-gpu partition can also allocate. Installing via conda also
# hits an unrelated MKL/ITT symbol mismatch on this cluster's image, so both
# branches use pip wheels from the official PyTorch index, not conda.
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
echo "Detected GPU compute capability: $COMPUTE_CAP"

if awk "BEGIN{exit !($COMPUTE_CAP >= 9.0)}"; then
    echo "Hopper or newer -> torch 2.1.2+cu118"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118
else
    echo "Volta/Ampere -> torch 1.13.0+cu117 (repo's documented pin)"
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117
fi

# requirements.txt pulls in torchtext==0.14.0, which hard-pins torch==1.13.0
# and silently downgrades whichever GPU build was just installed (pip warns
# about the conflict but does not fail). Reinstalling afterward, not before,
# is what actually matters here.
pip install -r requirements.txt
if awk "BEGIN{exit !($COMPUTE_CAP >= 9.0)}"; then
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118
fi

# flash_attn is a hard import in src/models/sequence/dna_embedding.py, even
# for the small from-scratch quick-start — there's no pure-PyTorch fallback
# in that code path. The pip wheel builds cleanly against whichever torch is
# now active; it is not a prebuilt binary wheel, so this step compiles.
pip install flash-attn --no-build-isolation

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
echo "Environment ready at $ENV_PREFIX"
