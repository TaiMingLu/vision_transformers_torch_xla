"""Smoke test for the TFDS-based ImageNet loader.

This script instantiates the `datasets.build_dataset` helper with arguments that
mirror the defaults in `main.py` and iterates through a few batches to verify
that samples can be produced from a TFDS installation on disk.

Example usage (do not run here, run on a machine with TFDS data installed)::

    python tools/test_tfds_loader.py \
        --data-dir /home/terry/gcs-bucket/Distillation/imagenet_tfds \
        --split train \
        --num-samples 4
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from datasets import build_dataset


def _build_args(data_dir: str,
                train_pp: str,
                eval_pp: str,
                seed: int,
                prefetch: int,
                num_parallel_calls: int,
                private_threadpool_size: int,
                shuffle_buffer: int) -> SimpleNamespace:
    """Create a namespace mirroring the attributes consumed by `build_dataset`."""

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
        world_size=1,
        rank=0,
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
                        help="Shuffle buffer size used when split=train")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    args = _build_args(
        data_dir=cli_args.data_dir,
        train_pp=cli_args.train_pp,
        eval_pp=cli_args.eval_pp,
        seed=cli_args.seed,
        prefetch=cli_args.prefetch,
        num_parallel_calls=cli_args.num_parallel_calls,
        private_threadpool_size=cli_args.private_threadpool_size,
        shuffle_buffer=cli_args.shuffle_buffer,
    )

    is_train = cli_args.split == "train"
    dataset, num_classes = build_dataset(is_train=is_train, args=args)
    print(f"Dataset built (split={cli_args.split}). num_classes={num_classes}")

    iterator = iter(dataset)
    for idx in range(cli_args.num_samples):
        image, label = next(iterator)
        assert isinstance(image, torch.Tensor) and isinstance(label, torch.Tensor)
        print(f"[{idx}] image shape={tuple(image.shape)} dtype={image.dtype} label={int(label)}")

    print("✅ TFDS loader smoke test completed successfully")


if __name__ == "__main__":
    main()
