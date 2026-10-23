# HyenaDNA — reproducible inference on PACE ICE

This README covers one task: loading a pretrained [HyenaDNA](https://github.com/HazyResearch/hyena-dna)
checkpoint and extracting DNA sequence embeddings, reproducibly, on Georgia
Tech's PACE ICE cluster (`ice-gpu` partition / AI Makerspace). Everything
here was run and verified on that cluster, not assumed from the upstream
docs — see [Troubleshooting](#troubleshooting) for the gaps found along the
way.

Model metadata: [`model.yaml`](model.yaml) · Execution recipe:
[`../../../../recipes/hyenadna-inference-embeddings.yaml`](../../../../recipes/hyenadna-inference-embeddings.yaml) ·
Resource profile: [`../../../../resource-profiles/pace-ice/ice-gpu.yaml`](../../../../resource-profiles/pace-ice/ice-gpu.yaml)

## Setup

```bash
ssh login-ice.pace.gatech.edu
srun --account=<your-account> --partition=ice-gpu --gres=gpu:1 \
     --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash

bash scripts/hyenadna_setup_env.sh
```

This clones `HazyResearch/hyena-dna` to `~/scratch/hyena-dna`, creates a
conda env at `~/scratch/.conda/envs/hyena-dna` (Python 3.8), and installs a
PyTorch build matched to whatever GPU Slurm actually allocated — see
[Hardware and provenance](#hardware-and-provenance) for why this matters.
Run it inside a GPU allocation, not on the login node: picking the right
PyTorch build depends on `nvidia-smi` seeing a real GPU.

## Execution

```bash
# smoke test + validation check (downloads the tiny-1k checkpoint on first run)
python scripts/hyenadna_validate.py --model hyenadna-tiny-1k-seqlen

# embed a sequence directly
python scripts/hyenadna_inference.py --model hyenadna-tiny-1k-seqlen \
    --sequence ACTGACTGACTGACTG

# or a FASTA file, saving the result
python scripts/hyenadna_inference.py --model hyenadna-tiny-1k-seqlen \
    --fasta data/example.fasta --output results/embeddings.pt
```

As a batch job:

```bash
sbatch scripts/hyenadna_inference.sbatch data/example.fasta
```

`scripts/hyenadna_inference.sbatch` builds the environment if it isn't
already there, runs the validation check, then the inference call — it is
the canonical, idempotent version of the steps above.

## Verification

`scripts/hyenadna_validate.py` is the validation check, run against the
`hyenadna-tiny-1k-seqlen` checkpoint on an ICE V100:

```
[smoke test] shape=(1, 202, 128) nan=False inf=False
[determinism] L2 distance across repeated runs of the same sequence: 0.00e+00
[validation] L2 distance between two unrelated sequences: 56.35 (embedding norm scale: 138.15)

VALIDATION PASSED
```

Four properties are checked, each catching a different failure mode a
shape-only smoke test would miss:

1. **Shape and finiteness** — output tensor shape matches `config["d_model"]`
   and the input length; no NaN/Inf (catches a broken forward pass).
2. **Determinism** — the same sequence embedded twice gives bit-identical
   output in eval mode (catches, e.g., an accidentally-active dropout layer).
3. **Input sensitivity** — two unrelated random sequences embed to
   measurably different vectors, by raw L2 distance on the un-pooled
   per-position embeddings (catches a checkpoint that loaded with the wrong
   keys, or silently fell back to random init, and so produces
   shape-correct but content-blind output).

4. **Pretraining objective** — the model predicts the next nucleotide, so a
   real human mRNA (NCBI `NM_000518.5`, HBB, at `data/example.fasta`) should get
   a higher mean next-nucleotide log-likelihood than a shuffle of the same
   nucleotides. The logits come from the tied embedding matrix, as in upstream
   training. A missing gap points at a loading or tokenization bug. On an
   H100 with the tiny-1k checkpoint, perplexity was 3.825 for the real
   sequence against 4.212 for the shuffle.

Cosine similarity on mean-pooled embeddings was tried first for check 3 and
rejected: it stayed above 0.99 even between two unrelated random sequences
for this checkpoint, so it couldn't actually discriminate anything. Raw
per-position L2 distance could.

## Hardware and provenance

- **Verified hardware:** PACE ICE `ice-gpu` partition, confirmed on a Tesla
  V100-16GB (compute capability 7.0). The setup script also handles Hopper
  GPUs (H100/H200, compute capability 9.0) that the same partition can
  allocate — confirmed separately, see below.
- **Checkpoint:** `LongSafari/hyenadna-tiny-1k-seqlen` from HuggingFace
  (BSD-3-Clause), downloaded via `git clone` (git-lfs) in
  `scripts/hyenadna_inference.py`.
- **Seeds:** the validation check uses fixed seeds (1, 2) for its two
  comparison sequences; inference on real input has no stochastic step
  (`model.eval()`, `torch.inference_mode()`), so no seed is needed there.
- **Software:** Python 3.8; PyTorch 1.13.0+cu117 or 2.1.2+cu118 depending on
  allocated GPU (see `scripts/hyenadna_setup_env.sh`); other pinned versions
  come from the upstream repo's `requirements.txt`.

## Troubleshooting

Three real gaps found while getting this to run, each worked around in the
scripts rather than just noted here:

**GPU architecture mismatch.** HyenaDNA's documented environment pins
`torch==1.13.0+cu117`, whose compiled kernels only cover compute capability
up to 8.6 (Ampere). PACE ICE's `ice-gpu` partition can also allocate H100 or
H200 nodes (compute capability 9.0), and on one of those, that pinned build
imports fine but hangs/spins at near-100% CPU on the first real GPU op,
trying to JIT a kernel for an architecture it doesn't ship. Fix:
`scripts/hyenadna_setup_env.sh` checks `nvidia-smi --query-gpu=compute_cap`
and installs `torch==2.1.2+cu118` instead when the GPU is Hopper or newer.

**`pip`'s resolver silently downgrades torch.** `requirements.txt` pulls in
`torchtext==0.14.0`, which hard-pins `torch==1.13.0`. Installing
`requirements.txt` after a GPU-enabled torch is already present downgrades
it back to a CPU-only-equivalent build — `pip` prints a dependency-conflict
warning but does not fail or stop. Fix: the setup script reinstalls the
correct GPU build of torch *after* `requirements.txt`, not before.

**Home directory quota breaks installs and downloads, not just training.**
PACE ICE's home quota (~30GB) is consumed by `pip`'s wheel cache,
HuggingFace's hub cache, and `git`'s global config file (written by
`git lfs install` on every checkpoint clone) well before scratch storage
fills up, and each one fails differently: `pip install` errors with `Disk
quota exceeded`; `git lfs` prints a non-fatal warning and continues (but the
clone can still fail depending on what else is pending); HuggingFace prints
a cache-write warning. Fix: `PIP_CACHE_DIR`, `TMPDIR`, `HF_HOME`, and
`GIT_CONFIG_GLOBAL` are all redirected to a scratch path before any install
or download step, in both the setup script and the sbatch job.

**Importing `huggingface.py` runs an inference example as a side effect.**
The upstream repo's documented pattern, `from huggingface import
HyenaDNAPreTrainedModel`, triggers a full hardcoded inference run at import
time — the module calls `inference_single()` unconditionally, with no `if
__name__ == "__main__":` guard. `scripts/hyenadna_inference.py` reimplements
the small amount of checkpoint-loading logic directly against
`standalone_hyenadna.py` instead of importing the upstream module. While
reimplementing it, a second bug surfaced: the upstream checkpoint-loading
code reads weights from `ckpt['state_dict']` — the raw `torch.load()` result
is a full PyTorch Lightning checkpoint dict (`epoch`, `global_step`,
`state_dict`, `optimizer_states`, ...), not a bare state dict — easy to miss
on a first reimplementation (it was missed here on the first attempt, and
caught by a `KeyError` rather than a silent wrong-weights load).

## Cleanup

```bash
rm -rf ~/scratch/hyena-dna/checkpoints   # downloaded weights (well under 1GB for tiny/small)
rm -rf ~/scratch/.conda/envs/hyena-dna   # conda environment
rm -rf ~/scratch/.cache/huggingface ~/scratch/.cache/pip  # redirected caches
```

The cloned repo at `~/scratch/hyena-dna` can stay — re-running
`scripts/hyenadna_setup_env.sh` is idempotent and skips the clone if it's
already there.
