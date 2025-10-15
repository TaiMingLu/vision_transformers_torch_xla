# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the TPU-first training entrypoint, delegating loops to `engine.py`, optimization defaults to `optim_factory.py`, and shared utilities to `utils/`. Model backbones sit in `models/` (timm-derived files plus TPU patches), while TFDS input plumbing is split between `datasets.py` and the bundled `big_vision/` ops. Use `tools/` for diagnostics and `train/` for launch helpers; keep experiment artefacts outside the repo to avoid noisy diffs.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: prepare a clean TPU-compatible environment.
- `pip install -r requirements.txt`: install torch-xla, timm, TFDS, and logging deps.
- `python main.py --help`: review CLI knobs for distillation, TFDS paths, and augmentation strings.
- `python tools/test_tfds_loader.py --data-dir <path> --split train --num-samples 8 --time-it`: sanity-check TFDS shards and preprocessing on CPU before a TPU run.
- `python -m torch_xla.distributed.xla_dist --tpu=$TPU -- python main.py --config ...`: mirror distributed launches shown in `run_commands.sh` when targeting TPU VMs.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions/modules, CamelCase for classes. Extend the existing type hints and dataclasses when touching public APIs, and keep imports grouped (stdlib, third-party, local). Use structured logging or `xm.rendezvous` status prints instead of ad-hoc `print` calls around TPU synchronization blocks.

## Testing Guidelines
Run `python test_kd.py` after modifying distillation or model-wrapping logic; it exercises student/teacher flows via `models.create_model`. Validate TFDS or preprocessing changes with the loader script above, ideally with the same splits you plan to train on. For regression checks, run short CPU/XLA passes (`--batch-size 8`) to trigger compile warmups deterministically, and add lightweight asserts near new helpers rather than growing `test_kd.py` indiscriminately.

## Commit & Pull Request Guidelines
History shows imperative, lower-case subjects (`fix`, `sync env`); keep that voice but add scope, e.g., `fix: guard tfds split parsing`. Squash noisy WIP commits locally so reviewers see one cohesive change. PR descriptions should capture TPU topology, data paths, env vars (`PJRT_DEVICE`, `XLA_USE_BF16`), and accuracy or loss deltas when touching training loops. Link issues or experiment trackers where possible.

## TPU & Data Configuration Tips
Always provide TFDS roots (`--tfds_data_dir` or `--data_path`) because the loader fails fast without them. When deploying to TPU VMs, follow `run_commands.sh`: reclone, recreate the venv, and reinstall wheels to avoid mismatched torch/xla builds. Confirm `HDF5_USE_FILE_LOCKING=FALSE` in shared environments to prevent GCS fuse deadlocks, and tweak preprocessing strings from `datasets.py` only when experiments truly require it.
