# HyenaDNA — reproducible pretrained inference on GT PACE ICE / AI Makerspace

This is the reproduction guide for running pretrained **HyenaDNA** inference on
Georgia Tech's ICE teaching cluster, including the "AI Makerspace" H100/H200
allocation. It is the first milestone of a semester plan that continues with
a downstream species-classification task and a reduced-scale pretraining
reproduction (see [Status & next steps](#status--next-steps)).

Authors: Kevin Yin ([@kyin68](https://github.com/kyin68)), [@Elliptic461](https://github.com/Elliptic461) — Georgia Institute of Technology.

---

## Contents

- [What this reproduces](#what-this-reproduces)
- [Provenance](#provenance)
- [The portability problem this recipe solves](#the-portability-problem-this-recipe-solves)
- [Repository layout](#repository-layout)
- [Reproducing it](#reproducing-it)
- [What the validation actually checks](#what-the-validation-actually-checks)
- [Execution evidence](#execution-evidence)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)
- [Status & next steps](#status--next-steps)

---

## What this reproduces

HyenaDNA is pretrained with a causal next-nucleotide (single-token) language
modeling objective on the human reference genome. This recipe:

1. Loads the smallest pretrained checkpoint (`hyenadna-tiny-1k-seqlen`) from
   its official Hugging Face source.
2. Runs it on a real human mRNA sequence (NCBI RefSeq `NM_000518.5`, HBB)
   and on a nucleotide-composition-matched random shuffle of that same
   sequence.
3. Produces per-position embeddings and reconstructs next-nucleotide
   log-likelihoods (the model's pretraining objective) for both.
4. Validates the result with a smoke test (did it run, is the output finite
   and correctly shaped) and a substantive scientific check (does the model
   actually find the real sequence more predictable than the scrambled one).

This is deliberately scoped to the smallest checkpoint so the whole loop
(env setup → inference → validation) runs in minutes on a single GPU of any
generation ICE might hand out. The same scripts work with the larger
checkpoints (`hyenadna-small-32k-seqlen` up to `hyenadna-large-1m-seqlen`) by
changing `--checkpoint`; see [Status & next steps](#status--next-steps) for
what that needs.

## Provenance

| Artifact | Source | Version / identifier |
|---|---|---|
| Model code | https://github.com/HazyResearch/hyena-dna | pinned commit `d553021b483b82980aa4b868b37ec2d4332e198a` (2023-12-03), cloned automatically by `environment/setup_env.sh` into `vendor/hyena-dna/` (not vendored into this repo — see [Cleanup](#cleanup)) |
| Paper | Nguyen, Poli, et al., *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution* | [arXiv:2306.15794](https://arxiv.org/abs/2306.15794) |
| Code license | Apache-2.0 | from upstream `LICENSE` |
| Pretrained checkpoint | `LongSafari/hyenadna-tiny-1k-seqlen` on Hugging Face, downloaded via `git lfs clone` inside `scripts/run_inference.py` | model card tags most HyenaDNA checkpoint sizes `bsd-3-clause`; **verify the license on each checkpoint's own HF model card before redistributing weights** — one secondary source lists `hyenadna-large-1m-seqlen` as a different, "custom" license, inconsistent with the smaller checkpoints, so don't assume it's the same without checking directly |
| Validation input | NCBI RefSeq `NM_000518.5`, *Homo sapiens* hemoglobin subunit beta (HBB) mRNA | committed at `data/NM_000518.5_HBB_mRNA.fasta`, retrieved via NCBI E-utilities (`efetch`, db=nuccore) on 2026-10-10. Committed rather than fetched at runtime because PACE ICE compute nodes may not have outbound access to `eutils.ncbi.nlm.nih.gov`. |
| Compute environment | GT PACE ICE, `ice-gpu` partition (includes the AI Makerspace H100/H200 allocation) | see [The portability problem this recipe solves](#the-portability-problem-this-recipe-solves) |

Model/recipe metadata for this entry also lives in this Foundry's registry format:
- `../model.yaml`
- `../recipes/inference-nextnucleotide.yaml`

## The portability problem this recipe solves

The upstream repo's documented install pins `torch==1.13.0` built against
CUDA 11.7 (`conda install pytorch==1.13.0 ... pytorch-cuda=11.7`). Those
wheels only ship compiled kernels through **sm_86** (Ampere). On ICE,
`sbatch --gres=gpu:1` on the `ice-gpu` partition can allocate a V100
(sm_70), A100 (sm_80), A40/RTX6000-Ada/L40S (sm_86/8.9), an **H100/H200**
(sm_90, the AI Makerspace hardware) — or even a non-CUDA AMD MI210 (ROCm)
node. The pinned build does not fail loudly on sm_90, it just doesn't have
a matching kernel image, which is a much harder failure mode to diagnose
than a clear version error.

We confirmed this directly: the documented pin would not run on an
ICE-allocated H100/H200 node, and switching to `torch==2.1.2+cu118` fixed it.

Rather than re-pinning to one new fixed version (which would just move the
problem to whichever future GPU generation ICE adds next), `environment/setup_env.sh`
**detects the allocated GPU's compute capability at job runtime**
(`nvidia-smi --query-gpu=compute_cap`) and picks the matching PyTorch/CUDA
build:

| Detected compute capability | GPU examples | PyTorch build installed |
|---|---|---|
| ≤ 8.6 | V100 (7.0), A100 (8.0), A40 (8.6) | `torch==1.13.0+cu117` (upstream's own pin) |
| > 8.6 | RTX 6000 Ada / L40S (8.9), H100 / H200 (9.0) | `torch==2.1.2+cu118` |
| no GPU detected | — (also what you'll see on the partition's AMD MI210 nodes, since `nvidia-smi` can't see them either) | CPU-only `torch` (with a warning; only use for a quick script-level smoke test, or request an NVIDIA-specific gres type to avoid landing on MI210) |

The ≤8.6 cutoff for the old pin is the officially documented one; the >8.6
branch is empirically confirmed on Hopper (sm_90) and extended by the same
reasoning to Ada (sm_89), since both postdate the cu117-era kernel builds.

This same hazard applies to *any* recipe pinned to an old CUDA wheel
(cu113/cu116/cu117) run on this partition — the fix generalizes beyond this
one model.

## Repository layout

```
registry/models/biology/hyenadna/
├── README.md                              <- you are here
├── model.yaml                             <- Foundry model registry entry
├── recipes/
│   └── inference-nextnucleotide.yaml      <- Foundry execution-recipe entry
├── environment/
│   └── setup_env.sh                       <- GPU-arch-aware env setup (the portability fix)
├── scripts/
│   ├── run_inference.py                   <- loads checkpoint, runs inference, saves outputs
│   └── validate_output.py                 <- smoke test + substantive validation check
├── data/
│   └── NM_000518.5_HBB_mRNA.fasta         <- real NCBI sequence used for validation
├── slurm/
│   └── infer_hyenadna_makerspace.sbatch   <- end-to-end Slurm batch job
├── logs/                                  <- execution logs land here (see Execution evidence)
├── results/                               <- created at run time: embeddings, log-likelihoods, validation_report.json
├── vendor/hyena-dna/                      <- created at run time: pinned upstream checkout (not committed)
└── checkpoints/                           <- created at run time: downloaded HF weights (not committed)
```

`vendor/`, `checkpoints/`, and `results/` are created by the scripts below
and are not committed to this repository — they're reproducible from the
pinned commit / checkpoint identifiers above, not large binary artifacts we
need to track in git.

## Reproducing it

### Prerequisites

- A Georgia Tech PACE ICE account with access to the `ice-gpu` partition
  (request through your course; see https://pace.gatech.edu/ice-cluster).
- SSH access (e.g. `ssh <user>@login-ice.pace.gatech.edu`, from campus
  network or GT VPN).
- `git`, `git-lfs`, and `conda` (via `module load anaconda3` on ICE). No manual
  CUDA toolkit install is needed — `environment/setup_env.sh` installs the
  correct PyTorch+CUDA wheel for whatever GPU you're given.

### Option A — one-shot batch job (recommended, gives you a clean log for free)

```bash
ssh <user>@login-ice.pace.gatech.edu
git clone -b kyin68/hyenadna-inference-reproduction https://github.com/Elliptic461/cybershuttle-ai4science-foundry.git
cd cybershuttle-ai4science-foundry/registry/models/biology/hyenadna
sbatch slurm/infer_hyenadna_makerspace.sbatch
# check status:
squeue -u $USER
# once COMPLETED, inspect:
cat logs/hyenadna-infer-<jobid>.out
cat results/validation_report.json
```

### Option B — interactive (useful while developing/debugging)

```bash
ssh <user>@login-ice.pace.gatech.edu
cd cybershuttle-ai4science-foundry/registry/models/biology/hyenadna
salloc --partition=ice-gpu --gres=gpu:h100:1 --cpus-per-task=4 --mem=32G --time=00:30:00
# once the allocation starts:
bash environment/setup_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hyenadna-infer

python scripts/run_inference.py \
  --checkpoint hyenadna-tiny-1k-seqlen \
  --input_fasta data/NM_000518.5_HBB_mRNA.fasta \
  --output_dir results \
  --device cuda

python scripts/validate_output.py \
  --results_dir results \
  --report results/validation_report.json
```

Either option ends with the same files in `results/`:
`embeddings_real.npy`, `embeddings_scrambled.npy`,
`loglik_real.npy`, `loglik_scrambled.npy`, `run_metadata.json`, and
`validation_report.json`.

## What the validation actually checks

**Smoke test** (`validate_output.py`): confirms every expected output file
exists, has the shape recorded in `run_metadata.json`, and contains only
finite values — i.e. the forward pass ran to completion and didn't silently
degrade into NaNs (a common failure mode when a GPU kernel mismatch or a
precision issue goes undetected).

**Substantive check**: HyenaDNA's pretraining objective is predicting the
next nucleotide. We reconstruct that objective's log-likelihood at
inference time (see the docstring in `scripts/run_inference.py` for exactly
how — it uses the tied embedding/output weight matrix, the same way the
upstream training code does) and compare the *real* HBB mRNA sequence
against a random shuffle of the identical nucleotides. If the pretrained
weights encode real genomic structure, the real sequence should get a
measurably higher mean log-likelihood (lower perplexity) than its scrambled
twin — same base composition, different order. `validate_output.py` reports
both perplexities and the gap, and exits non-zero if the gap isn't positive,
which would point to a loading/tokenization bug rather than a genuine model
limitation (even a lightly-trained genomic LM should show *some* gap on a
real coding mRNA).

This is intentionally a correctness check, not a benchmark claim — it tests
"did this reproduction actually load and run the real pretrained model,"
not "does this match a published accuracy number."

## Execution evidence

Each run leaves:
- `logs/hyenadna-infer-<jobid>.out` / `.err` — the full Slurm job log (GPU
  detected, torch build installed, per-sequence embedding shapes,
  perplexities, wall time) for an ICE/AI Makerspace run; or a local run log
  for a CPU pre-flight check.
- `results/run_metadata.json` — checkpoint id, device, torch/CUDA versions,
  per-sequence metrics.
- `results/validation_report.json` — smoke test + substantive check result.

Before running on ICE, we validated the pipeline **locally on CPU** (same
scripts, same checkpoint, same input, just `--device cpu`) to catch bugs
cheaply — see `logs/local-cpu-dryrun/`. That run: completed both forward
passes with finite, correctly-shaped output (smoke test **PASS**), and gave
the real HBB mRNA sequence a mean next-nucleotide log-likelihood of
**-1.3416 nats** (perplexity 3.825) vs. **-1.4380 nats** (perplexity 4.212)
for the scrambled control — a +0.0964 nat gap in the expected direction
(substantive check **PASS**). It also caught two real missing dependencies
(`torchvision`, `omegaconf`) and a PyTorch ≥2.6 `weights_only` incompatibility,
now fixed — see Troubleshooting below.

**The actual ICE / AI Makerspace GPU run** completed successfully — see
`logs/ice-h100-makerspace/`. Submitted via `sbatch slurm/infer_hyenadna_makerspace.sbatch`,
it was allocated a real NVIDIA H100 80GB HBM3 (compute capability 9.0,
confirmed via the job log's `nvidia-smi` output), correctly selected
`torch==2.1.2+cu118` per the GPU-detection logic, and ran end to end in
29 seconds (job 6188466; walltime from the Slurm epilog). Result: smoke
test **PASS**; substantive check **PASS** with the identical perplexities
as the local CPU run (3.825 real vs. 4.212 scrambled) — confirming
deterministic, correct inference on the actual target AI Makerspace
hardware, not just on CPU.

Getting there took three submissions. The first two failed for real,
non-obvious environment reasons (a `conda activate` / `set -u` conflict,
then a PyTorch/NumPy version incompatibility) — both are fixed in
`environment/setup_env.sh` and documented in Troubleshooting below, and the
failed attempts themselves are described in
`logs/ice-h100-makerspace/README.md`.

## Troubleshooting

- **`nvidia-smi: command not found` / no GPU detected in `setup_env.sh`** —
  you're on the ICE login node, not inside a GPU allocation (or you landed
  on one of the partition's AMD MI210 nodes, which `nvidia-smi` can't see).
  Use `sbatch` or `salloc --gres=gpu:<type>:1` with an NVIDIA-specific type.
- **Checkpoint download fails / hangs** — `git-lfs` isn't installed or
  `module load git-lfs` didn't run. Check with `git lfs version`; the
  compute node also needs outbound HTTPS access to `huggingface.co`,
  which should be allowed on ICE but confirm if this is your first time
  using it there.
- **`CUDA error: no kernel image is available for execution on the device`**
  — this is exactly the sm_86-ceiling problem this recipe is built to avoid.
  If you see it, `environment/setup_env.sh` either wasn't run, was run on a
  different node than the job actually executes on, or its compute-capability
  branch needs updating for a new GPU generation — check
  `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` against the
  table above.
- **`KeyError: key mismatch in the state dicts!` in `run_inference.py`** —
  usually means the config/checkpoint pair don't match (e.g. passing a
  different checkpoint name than what was downloaded into `checkpoints/`).
  Delete `checkpoints/<name>/` and re-run to force a clean re-download.
- **`ModuleNotFoundError: No module named 'torchvision'`** —
  `standalone_hyenadna.py` imports `torchvision.ops.StochasticDepth` at
  module load time even though inference never calls it.
  `environment/setup_env.sh` installs `torchvision` alongside `torch` in
  every branch; if you're not using that script, install it manually.
- **`ModuleNotFoundError: No module named 'omegaconf'`** —
  `weights.ckpt` is a PyTorch Lightning checkpoint with a pickled
  `omegaconf.listconfig.ListConfig` object saved alongside the tensors;
  unpickling it needs the class importable even though `run_inference.py`
  never reads its contents. `pip install omegaconf` (already in
  `environment/setup_env.sh`).
- **`_pickle.UnpicklingError: Weights only load failed`** — on PyTorch
  ≥2.6, `torch.load` defaults to `weights_only=True`, which rejects this
  checkpoint for the same OmegaConf-object reason above.
  `scripts/run_inference.py` already passes `weights_only=False`
  explicitly (safe here since the checkpoint is from the official,
  documented LongSafari source — see `model.yaml`'s `upstream` field).
- **Job exceeds the 30-minute time limit** — only likely if the checkpoint
  download is slow on a cold HF cache; re-running reuses the cached
  download in `checkpoints/`, so a retry after a timeout is fast.
- **`import torch` crashes with "A module that was compiled using NumPy 1.x
  cannot be run in NumPy 2.x"`** — hit this ourselves on the first real ICE
  H100 run. `torch==2.1.2` (the Hopper-compatible pin) predates NumPy 2.0
  and breaks if an unpinned `pip install numpy` grabs the latest 2.x.
  `environment/setup_env.sh` now pins `"numpy<2"`.
- **Job dies silently right after `conda create`, with only an
  `unbound variable` line in the `.err` file** — hit this ourselves on the
  first real ICE run. PACE's miniforge install ships conda activate/deactivate
  hook scripts (e.g. `libxml2_deactivate.sh`) that reference an unset
  variable; under this script's `set -euo pipefail`, that's fatal. Both
  `environment/setup_env.sh` and the sbatch script now wrap their
  `conda activate` calls in `set +u` / `set -u` to tolerate it — if you've
  copied the `conda activate` lines elsewhere, carry that wrapping with them.

## Cleanup

Everything this recipe creates is reproducible from pinned identifiers, so
it's safe to delete to reclaim ICE storage quota:

```bash
conda env remove -n hyenadna-infer
rm -rf vendor/hyena-dna checkpoints results
# keep logs/ and data/ — those are the provenance/evidence record
```

## Status & next steps

- [x] Pretrained HyenaDNA inference reproduced on PACE ICE with GPU-architecture
      auto-detection (this entry).
- [ ] Downstream task using HyenaDNA embeddings (e.g. species-of-origin
      classification from genomic sequence).
- [ ] Reduced-scale pretraining reproduction (shorter context length,
      smaller model) to study the training loop itself.
- [ ] Validate the same recipe on Purdue Anvil to confirm the GPU-detection
      approach generalizes beyond ICE-specific quirks.
