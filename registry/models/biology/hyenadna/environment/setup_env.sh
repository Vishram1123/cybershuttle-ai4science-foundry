#!/usr/bin/env bash
# Reproducible environment setup for pretrained HyenaDNA inference.
#
# The upstream repo (https://github.com/HazyResearch/hyena-dna) pins
# torch==1.13.0 built against cu117. Those wheels only ship compiled kernels
# through sm_86 (Ampere). On GT PACE ICE's ice-gpu partition / AI Makerspace,
# `--gres=gpu:1` can land you on a V100 (sm_70), A100 (sm_80), or an
# H100/H200 (sm_90) node — and on sm_90 the pinned build silently fails (no
# matching kernel image) instead of erroring clearly. This script detects the
# allocated GPU's compute capability at job runtime and installs a matching
# PyTorch/CUDA build instead of hardcoding one, so the same recipe works
# across whichever node Slurm hands out.
#
# Usage: run this INSIDE an active Slurm allocation (sbatch job or an
# `srun --pty` / `salloc` session) that already has a GPU, not on the login
# node — GPU detection requires the allocated GPU to be visible.
#
# Idempotent: safe to re-run; recreates the conda env if it already exists.

set -euo pipefail

ENV_NAME="hyenadna-infer"
PYTHON_VERSION="3.10"
HYENA_DNA_COMMIT="d553021b483b82980aa4b868b37ec2d4332e198a"  # pinned upstream commit (2023-12-03)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/../vendor/hyena-dna"

echo "== HyenaDNA inference environment setup =="

# --- 0. Make conda available (PACE ICE uses Lmod modules; no-op elsewhere) ---
if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    echo "conda not on PATH; trying 'module load anaconda3' (PACE ICE)"
    module load anaconda3 || true
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available. On PACE ICE run 'module load anaconda3' first." >&2
  exit 1
fi

# --- 1. Detect the allocated GPU's compute capability ---
GPU_CC=""
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
  if [[ -n "${GPU_LINE}" ]]; then
    GPU_NAME="$(echo "${GPU_LINE}" | cut -d',' -f1 | sed 's/^ *//;s/ *$//')"
    GPU_CC="$(echo "${GPU_LINE}" | cut -d',' -f2 | sed 's/^ *//;s/ *$//')"
  fi
fi

if [[ -z "${GPU_CC}" ]]; then
  echo "WARNING: no GPU detected (nvidia-smi missing or returned nothing)."
  echo "         Falling back to CPU-only torch. Make sure you requested"
  echo "         --gres=gpu:<type>:1 and are running this inside the"
  echo "         allocation, not on the ICE login node. Note: this also"
  echo "         happens correctly-but-unhelpfully on the partition's AMD"
  echo "         MI210 (ROCm) nodes, which nvidia-smi can't see either —"
  echo "         request an NVIDIA-specific gres type to avoid landing there."
  TORCH_SPEC="torch torchvision"
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  echo "Detected GPU: ${GPU_NAME} (compute capability ${GPU_CC})"
  # Compare as a float. sm_86 (Ampere, cc 8.0-8.6) is the documented ceiling
  # for the upstream torch==1.13.0+cu117 pin. cc 8.9 (Ada: L40/RTX6000 Ada)
  # and 9.0 (Hopper: H100/H200) need a newer build; 9.0 is the case we hit
  # and confirmed empirically on an ICE H100 allocation.
  CC_OK_FOR_OLD_PIN=$(awk -v cc="${GPU_CC}" 'BEGIN { print (cc <= 8.6) ? "1" : "0" }')
  if [[ "${CC_OK_FOR_OLD_PIN}" == "1" ]]; then
    echo "-> compute capability <= 8.6: using upstream pin torch==1.13.0+cu117"
    TORCH_SPEC="torch==1.13.0+cu117 torchvision==0.14.0+cu117"
    TORCH_INDEX="https://download.pytorch.org/whl/cu117"
  else
    echo "-> compute capability > 8.6 (Ada/Hopper): upstream pin lacks kernels for this GPU."
    echo "   using torch==2.1.2+cu118 instead (verified working on H100/H200)."
    TORCH_SPEC="torch==2.1.2+cu118 torchvision==0.16.2+cu118"
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
  fi
fi

# --- 2. Create / recreate the conda env ---
if conda env list | grep -qE "^${ENV_NAME} "; then
  echo "Conda env '${ENV_NAME}' already exists; reusing it."
else
  echo "Creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

# conda's own activate/deactivate hook scripts (e.g. PACE's miniforge 24.3.0
# ships a libxml2_deactivate.sh that references an unset variable) aren't
# written to be `set -u`-safe, and will kill this script under nounset with
# an opaque "unbound variable" error. Relax -u only around conda's own code.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

# --- 3. Install torch (GPU-arch-matched, or CPU fallback) ---
pip install ${TORCH_SPEC} --index-url "${TORCH_INDEX}"

# --- 4. Install the remaining inference-only dependencies ---
# No flash-attention / pykeops build needed (those are for training-time fused-kernel
# speedups and custom S4D layers, not pretrained inference). torchvision is required
# only because standalone_hyenadna.py imports torchvision.ops.StochasticDepth at module
# load time, even though inference never calls it. omegaconf is required because the
# checkpoint's weights.ckpt is a PyTorch Lightning checkpoint with a pickled OmegaConf
# config object alongside the tensors — unpickling needs the class importable, even
# though we never read its contents. numpy<2 is pinned because torch==2.1.2 (Dec 2023)
# predates NumPy 2.0 (June 2024) and crashes on `import torch` if an unpinned `pip
# install numpy` pulls in numpy>=2 — "A module that was compiled using NumPy 1.x cannot
# be run in NumPy 2.x" — hit this for real on an ICE H100 run, not found by reading docs.
pip install "transformers==4.26.1" einops "numpy<2" omegaconf

# --- 5. Fetch the pinned upstream commit (for standalone_hyenadna.py / huggingface.py) ---
if [[ -d "${VENDOR_DIR}/.git" ]]; then
  echo "Upstream hyena-dna checkout already present at ${VENDOR_DIR}"
else
  mkdir -p "$(dirname "${VENDOR_DIR}")"
  echo "Cloning HazyResearch/hyena-dna @ ${HYENA_DNA_COMMIT}"
  git clone https://github.com/HazyResearch/hyena-dna.git "${VENDOR_DIR}"
fi
git -C "${VENDOR_DIR}" fetch --depth 1 origin "${HYENA_DNA_COMMIT}" 2>/dev/null || true
git -C "${VENDOR_DIR}" checkout "${HYENA_DNA_COMMIT}"

# --- 6. git-lfs is required to pull checkpoint weights from the LongSafari HF repos ---
if ! command -v git-lfs >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load git-lfs 2>/dev/null || true
  fi
fi
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "WARNING: git-lfs not found. Checkpoint download in scripts/run_inference.py"
  echo "         will fail. On PACE ICE try 'module load git-lfs', or"
  echo "         'conda install -y -c conda-forge git-lfs'."
fi

echo "== Setup complete =="
echo "conda env:        ${ENV_NAME}"
echo "python:           $(python --version)"
echo "torch:             $(python -c 'import torch; print(torch.__version__, "cuda:", torch.version.cuda)' 2>/dev/null || echo 'import failed')"
echo "cuda available:    $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo "hyena-dna checkout: ${VENDOR_DIR} @ $(git -C "${VENDOR_DIR}" rev-parse --short HEAD)"
