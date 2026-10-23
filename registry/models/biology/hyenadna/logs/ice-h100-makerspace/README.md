# ICE / AI Makerspace H100 run (job 6188466)

Real execution on GT PACE ICE's `ice-gpu` partition, allocated an NVIDIA H100
80GB HBM3 (AI Makerspace hardware, compute capability 9.0), via
`slurm/infer_hyenadna_makerspace.sbatch`.

- Node: `atl1-1-03-012-23-0`
- Walltime: 29s (torch/env already cached from two earlier attempts on the same node/scratch dir — see below)
- Result: smoke test PASS, substantive validation PASS — real HBB mRNA perplexity 3.825 vs. 4.212 for the scrambled control, identical to the local CPU dry run (`../local-cpu-dryrun/`), confirming deterministic, correct inference on the actual target hardware.

This was the third submission (job IDs 6188372, 6188411, 6188466). The first
two failed and are not included here, but are documented in
`../../README.md`'s Troubleshooting section since both were real,
non-obvious bugs fixed in `environment/setup_env.sh`:

1. **6188372** — died right after `conda create`/`conda activate`: PACE's
   conda ships activate/deactivate hooks that reference an unset shell
   variable, fatal under this script's `set -euo pipefail`. Fixed by
   wrapping `conda activate` calls in `set +u` / `set -u`.
2. **6188411** — got past that, correctly detected the H100 and selected
   `torch==2.1.2+cu118`, but crashed on `import torch` because an unpinned
   `pip install numpy` pulled NumPy 2.2.6, which `torch==2.1.2` (predates
   NumPy 2.0) can't load. Fixed by pinning `"numpy<2"`.

Both fixes are environment/portability bugs specific to running this on a
real, shared teaching cluster — exactly the kind of thing a from-scratch
local test can't surface, which is why this ICE run mattered even after the
local CPU dry run had already passed.
