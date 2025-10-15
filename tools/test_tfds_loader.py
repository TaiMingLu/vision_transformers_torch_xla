"""Smoke test for the TFDS-based ImageNet loader.

This script mirrors the defaults in ``main.py`` to verify that the vendored
Big Vision preprocessing stack and TFDS sharding behave as expected. It can be
run in single-process mode or with manually supplied ``--world-size`` /
``--rank`` values to emulate distributed launches.

Example (single process)::

    python tools/test_tfds_loader.py \
        --data-dir /home/terry/gcs-bucket/Distillation/imagenet_tfds \
        --split train \
        --num-samples 8

To mimic an 8-way setup you can run the script multiple times varying
``--rank`` (0..7) with ``--world-size 8``. When launched through
``torch_xla.distributed.xla_dist`` or PJRT, those values are filled by the
training harness so no overrides are needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

# Ensure we import the local `datasets` module instead of Hugging Face's package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402


def _build_args(data_dir: str,
                train_pp: str,
                eval_pp: str,
                seed: int,
                prefetch: int,
                num_parallel_calls: int,
                private_threadpool_size: int,
                shuffle_buffer: int,
                world_size: int,
                rank: int) -> SimpleNamespace:
    """Create a namespace mirroring the attributes consumed by ``build_dataset``."""

    return SimpleNamespace(
        data_set="imagenet2012",
        data_path=data_dir,
        eval_data_path=data_dir,
        tfds_data_dir=data_dir,
        tfds_train_split="train",
        tfds_eval_split="validation",
        tfds_shuffle_buffer=shuffle_buffer,
        tfds_cache_raw=False,
        tfds_cache_eval=False,
        tfds_prefetch=prefetch,
        tfds_num_parallel_calls=num_parallel_calls,
        tfds_private_threadpool_size=private_threadpool_size,
        tfds_skip_decode=True,
        big_vision_pp_train=train_pp,
        big_vision_pp_eval=eval_pp,
        big_vision_normalize="imagenet",
        seed=seed,
        world_size=world_size,
        rank=rank,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the TFDS ImageNet loader")
    parser.add_argument("--data-dir", required=True,
                        help="Root directory containing TFDS ImageNet shards (e.g. imagenet2012/5.1.0)")
    parser.add_argument("--split", default="train", choices=["train", "validation"],
                        help="TFDS split to iterate")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of samples to draw for the smoke test")
    parser.add_argument("--train-pp", default="decode_jpeg_and_inception_crop(224)|flip_lr|value_range(0, 1)|keep(\"image\", \"label\")",
                        help="big_vision preprocessing string for training")
    parser.add_argument("--eval-pp", default="decode|resize_small(256)|central_crop(224)|value_range(0, 1)|keep(\"image\", \"label\")",
                        help="big_vision preprocessing string for evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed fed to the TF pipeline")
    parser.add_argument("--prefetch", type=int, default=2, help="tf.data prefetch depth")
    parser.add_argument("--num-parallel-calls", type=int, default=100,
                        help="Number of parallel calls for preprocessing")
    parser.add_argument("--private-threadpool-size", type=int, default=48,
                        help="Threadpool size for tf.data host workers")
    parser.add_argument("--shuffle-buffer", type=int, default=250_000,
                        help="Shuffle buffer size used when split=train (set 0 for eval)")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Total number of processes participating in the data pipeline")
    parser.add_argument("--rank", type=int, default=0,
                        help="This process's rank (0-indexed)")
    parser.add_argument("--time-it", action="store_true",
                        help="Print rough throughput timing for the requested samples")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()

    if cli_args.rank < 0 or cli_args.rank >= cli_args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")

    args = _build_args(
        data_dir=cli_args.data_dir,
        train_pp=cli_args.train_pp,
        eval_pp=cli_args.eval_pp,
        seed=cli_args.seed,
        prefetch=cli_args.prefetch,
        num_parallel_calls=cli_args.num_parallel_calls,
        private_threadpool_size=cli_args.private_threadpool_size,
        shuffle_buffer=cli_args.shuffle_buffer if cli_args.split == "train" else 0,
        world_size=cli_args.world_size,
        rank=cli_args.rank,
    )

    is_train = cli_args.split == "train"
    dataset, num_classes = build_dataset(is_train=is_train, args=args)
    print(f"Dataset built (split={cli_args.split}) rank={cli_args.rank}/{cli_args.world_size}. "
          f"num_classes={num_classes}")

    iterator = iter(dataset)
    start = perf_counter() if cli_args.time_it else None

    for idx in range(cli_args.num_samples):
        image, label = next(iterator)
        assert isinstance(image, torch.Tensor) and isinstance(label, torch.Tensor)
        print(f"[{idx}] image shape={tuple(image.shape)} dtype={image.dtype} label={int(label)}")

    if start is not None:
        elapsed = perf_counter() - start
        print(f"Timing: fetched {cli_args.num_samples} samples in {elapsed:.2f}s"
              f" ⇒ {(cli_args.num_samples / max(elapsed, 1e-6)):.2f} samples/s")

    print("✅ TFDS loader smoke test completed successfully")


if __name__ == "__main__":
    main()
