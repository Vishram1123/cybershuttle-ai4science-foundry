# Local CPU pre-flight dry run

Run before the ICE/AI Makerspace GPU execution, to validate the recipe
itself (scripts, checkpoint loading, validation logic) without waiting on
cluster access. Same pipeline as the real run, just `--device cpu` and a
plain Python 3.10 venv instead of `environment/setup_env.sh`'s conda/GPU path.

Files:
- `run.log` — full terminal output of both scripts
- `run_metadata.json` — per-sequence metrics (checkpoint, device, perplexities)
- `validation_report.json` — smoke test + substantive validation result (both PASS)

This caught two real missing dependencies (`torchvision`, `omegaconf`) and a
PyTorch >=2.6 `weights_only` incompatibility, now fixed in
`environment/setup_env.sh` / `scripts/run_inference.py` — see the
Troubleshooting section in `../../README.md`.
