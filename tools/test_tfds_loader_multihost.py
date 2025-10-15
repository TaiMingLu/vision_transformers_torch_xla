"""Stress-test and validate the TFDS ImageNet loader in multi-host setups.

This script iterates the Big Vision-based TFDS input pipeline for multiple
loops per worker, checking two invariants:

1. Sharding correctness — every TFDS example id observed across all ranks must
   be unique.
2. Throughput stability — per-rank and global throughput must not degrade beyond
   the configured ratio or minimum samples/sec threshold across iterations.

Run with PJRT/TPU via ``torch_xla.distributed.xla_dist`` so that multi-host
process orchestration and TPU topology env vars are populated correctly.
Example::

    python -m torch_xla.distributed.xla_dist --tpu=$TPU_NAME -- \
        python tools/test_tfds_loader_multihost.py \
            --data-dir /home/terry/gcs-bucket/Distillation/imagenet_tfds \
            --split train --samples-per-loop 128 --num-loops 32 --log-every 4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import sysconfig
import torch

DEFAULT_TPU_LIBRARY_CANDIDATES = (
    "/lib/libtpu.so",
    "/lib/x86_64-linux-gnu/libtpu.so",
    "/usr/lib/libtpu.so",
    "/usr/lib/x86_64-linux-gnu/libtpu.so",
    "/usr/local/lib/libtpu.so",
)


def _ensure_tfds_id(pipeline: str) -> str:
    """Ensure the preprocessing pipeline keeps tfds_id."""

    if "tfds_id" in pipeline:
        return pipeline

    def _inject(match: re.Match[str]) -> str:
        contents = match.group(1)
        if "tfds_id" in contents:
            return match.group(0)
        suffix = ", \"tfds_id\"" if contents.strip() else "\"tfds_id\""
        return f"keep({contents}{suffix})"

    updated, count = re.subn(r"keep\(([^)]*)\)", _inject, pipeline, count=1)
    if count:
        return updated

    return pipeline + "|keep(\"image\", \"label\", \"tfds_id\")"


def _libtpu_candidates() -> List[Path]:
    """Enumerate likely libtpu locations on TPU VM images and wheels."""

    candidates = [Path(path) for path in DEFAULT_TPU_LIBRARY_CANDIDATES]

    env_roots = [
        os.environ.get("LIBTPU_ROOT"),
        os.environ.get("TPU_LIBRARY_PATH"),
    ]
    for root in env_roots:
        if root:
            root_path = Path(root)
            candidates.append(root_path)
            candidates.append(root_path / "libtpu.so")

    try:  # site packages contain the pip-installed libtpu wheel
        import site

        site_dirs = list(site.getsitepackages())
        user_dir = site.getusersitepackages()
        if user_dir:
            site_dirs.append(user_dir)
        for directory in site_dirs:
            dir_path = Path(directory)
            candidates.append(dir_path / "libtpu.so")
            candidates.append(dir_path / "libtpu" / "libtpu.so")
    except (ImportError, AttributeError):
        pass

    sys_paths = {
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sysconfig.get_config_var("LIBDIR") or ""),
        Path(sysconfig.get_config_var("LIBPL") or ""),
    }
    for base in sys_paths:
        if not base:
            continue
        candidates.append(base / "libtpu.so")
        candidates.append(base / "lib" / "libtpu.so")
        candidates.append(base / "lib64" / "libtpu.so")
        candidates.append(base / "libtpu" / "libtpu.so")

    spec = importlib.util.find_spec("libtpu")
    if spec and spec.origin:
        package_dir = Path(spec.origin).parent
        candidates.append(package_dir / "libtpu.so")
        candidates.append(package_dir / "libtpu" / "libtpu.so")

    try:
        import ctypes.util

        located = ctypes.util.find_library("tpu")
        if located:
            if os.path.isabs(located):
                candidates.append(Path(located))
            else:
                candidates.append(Path("/usr/lib") / located)
                candidates.append(Path("/usr/local/lib") / located)
    except ImportError:
        pass

    return candidates


def _maybe_configure_pjrt_library() -> None:
    """Populate PJRT_TPU_LIBRARY_PATH when running against TPU PJRT."""

    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        return
    if os.environ.get("PJRT_TPU_LIBRARY_PATH"):
        return

    import ctypes

    for candidate in _libtpu_candidates():
        if not candidate:
            continue
        path = candidate
        if path.is_dir():
            path = path / "libtpu.so"
        if not path.is_file():
            continue
        try:
            ctypes.CDLL(str(path))
        except OSError as exc:
            print(f"warning: skipping {path} (failed to load libtpu.so: {exc})", flush=True)
            continue
        os.environ["PJRT_TPU_LIBRARY_PATH"] = str(path)
        print(f"Auto-configured PJRT_TPU_LIBRARY_PATH={path}", flush=True)
        return

    raise SystemExit(
        "error: could not locate a usable libtpu.so. Set PJRT_TPU_LIBRARY_PATH to the "
        "full path of libtpu.so on the TPU VM (for example /lib/x86_64-linux-gnu/libtpu.so)."
    )


_maybe_configure_pjrt_library()

DEFAULT_TFDS_DIR = os.environ.get(
    "VISION_TFDS_DIR", "/home/terry/gcs-bucket/Distillation/imagenet_tfds")

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
except ImportError as exc:  # pragma: no cover - TPU-only helper
    raise ImportError(
        "torch_xla must be installed to run the multihost TFDS loader test") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS_PATH = REPO_ROOT / "datasets.py"
if not DATASETS_PATH.exists():
    raise FileNotFoundError(f"Could not find datasets.py at {DATASETS_PATH}")

DATASETS_SPEC = importlib.util.spec_from_file_location("vision_datasets", DATASETS_PATH)
assert DATASETS_SPEC and DATASETS_SPEC.loader, "Failed to build import spec for datasets.py"
vision_datasets = importlib.util.module_from_spec(DATASETS_SPEC)
sys.modules[DATASETS_SPEC.name] = vision_datasets
DATASETS_SPEC.loader.exec_module(vision_datasets)
build_dataset = vision_datasets.build_dataset


def _env_default(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def _concat_lists(left: List[str], right: List[str]) -> List[str]:
    return list(left) + list(right)


def _build_args(data_dir: str,
                train_pp: str,
                eval_pp: str,
                seed: int,
                prefetch: int,
                num_parallel_calls: int,
                private_threadpool_size: int,
                shuffle_buffer: int,
                world_size: int,
                rank: int) -> argparse.Namespace:
    train_pp = _ensure_tfds_id(train_pp)
    eval_pp = _ensure_tfds_id(eval_pp)

    namespace = argparse.Namespace(
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
        tfds_return_id=True,
    )
    return namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multihost TFDS loader stress test")
    parser.add_argument("--data-dir", default=DEFAULT_TFDS_DIR,
                        help=("Root directory containing TFDS ImageNet shards."
                              " Defaults to $VISION_TFDS_DIR or /home/terry/gcs-bucket/Distillation/imagenet_tfds."))
    parser.add_argument("--split", default="train", choices=["train", "validation"],
                        help="TFDS split to iterate")
    parser.add_argument("--samples-per-loop", type=int, default=128,
                        help="Samples drawn per loop on each worker")
    parser.add_argument("--num-loops", type=int, default=16,
                        help="Number of loops to execute per worker")
    parser.add_argument("--train-pp", default="decode_jpeg_and_inception_crop(224)|flip_lr|value_range(0, 1)|keep(\"image\", \"label\", \"tfds_id\")",
                        help="big_vision preprocessing string for training")
    parser.add_argument("--eval-pp", default="decode|resize_small(256)|central_crop(224)|value_range(0, 1)|keep(\"image\", \"label\", \"tfds_id\")",
                        help="big_vision preprocessing string for evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for TFDS shuffling")
    parser.add_argument("--prefetch", type=int, default=2, help="tf.data prefetch depth")
    parser.add_argument("--num-parallel-calls", type=int, default=100,
                        help="tf.data parallel call count")
    parser.add_argument("--private-threadpool-size", type=int, default=48,
                        help="Private threadpool size for tf.data host workers")
    parser.add_argument("--shuffle-buffer", type=int, default=250_000,
                        help="Shuffle buffer when split=train (unused for eval)")
    parser.add_argument("--world-size", type=int, default=None,
                        help="Optional override for world size")
    parser.add_argument("--rank", type=int, default=None,
                        help="Optional override for rank")
    parser.add_argument("--min-throughput-ratio", type=float, default=0.6,
                        help="Minimum allowed ratio of min/max throughput per rank")
    parser.add_argument("--min-samples-per-sec", type=float, default=0.5,
                        help="Minimum per-rank throughput in samples/sec (set <=0 to disable)")
    parser.add_argument("--log-every", type=int, default=4,
                        help="How often (in loops) to log per-rank throughput")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()

    if not cli_args.data_dir:
        raise SystemExit("error: specify --data-dir or set VISION_TFDS_DIR to point at the TFDS root")

    if cli_args.world_size is not None:
        world_size = cli_args.world_size
    else:
        try:
            world_size = xr.world_size()
        except RuntimeError:
            world_size = _env_default("WORLD_SIZE", _env_default("NUM_PROCESSES", 1))

    if cli_args.rank is not None:
        rank = cli_args.rank
    else:
        try:
            rank = xr.global_ordinal()
        except RuntimeError:
            rank = _env_default("RANK", _env_default("PROCESS_INDEX", 0))

    if rank < 0 or rank >= max(world_size, 1):
        raise ValueError("rank must satisfy 0 <= rank < world_size")

    is_train = cli_args.split == "train"
    shuffle_buffer = cli_args.shuffle_buffer if is_train else 0

    args = _build_args(
        data_dir=cli_args.data_dir,
        train_pp=cli_args.train_pp,
        eval_pp=cli_args.eval_pp,
        seed=cli_args.seed,
        prefetch=cli_args.prefetch,
        num_parallel_calls=cli_args.num_parallel_calls,
        private_threadpool_size=cli_args.private_threadpool_size,
        shuffle_buffer=shuffle_buffer,
        world_size=world_size,
        rank=rank,
    )

    dataset, num_classes = build_dataset(is_train=is_train, args=args)
    print(f"Dataset ready (split={cli_args.split}) rank={rank}/{world_size} num_classes={num_classes}", flush=True)

    iterator = iter(dataset)
    local_ids: List[str] = []
    local_throughputs: List[float] = []
    local_seen: set[str] = set()
    local_id_hashes: List[int] = []

    total_samples_per_rank = cli_args.samples_per_loop * cli_args.num_loops

    for loop_idx in range(cli_args.num_loops):
        start = perf_counter()
        loop_ids: List[str] = []
        for _ in range(cli_args.samples_per_loop):
            sample = next(iterator)
            if len(sample) != 3:
                raise RuntimeError(
                    "Dataset did not return (image, label, tfds_id). Ensure tfds_return_id is enabled")
            _image, _label, tfds_id = sample
            if not isinstance(tfds_id, str):
                raise TypeError(f"Expected tfds_id to be str, got {type(tfds_id)!r}")
            loop_ids.append(tfds_id)
            id_digest = hashlib.blake2b(
                tfds_id.encode('utf-8'), digest_size=8).digest()
            local_id_hashes.append(int.from_bytes(id_digest, 'big', signed=True))
        elapsed = perf_counter() - start
        throughput = cli_args.samples_per_loop / max(elapsed, 1e-6)

        loop_set = set(loop_ids)
        if len(loop_set) != len(loop_ids):
            raise RuntimeError(
                f"Rank {rank} observed duplicate TFDS ids within loop {loop_idx}")
        overlap = local_seen.intersection(loop_set)
        if overlap:
            dup_preview = ', '.join(sorted(list(overlap))[:3])
            raise RuntimeError(
                f"Rank {rank} observed repeated TFDS ids across loops: {dup_preview}")
        local_seen.update(loop_set)

        local_ids.extend(loop_ids)
        local_throughputs.append(throughput)

        if cli_args.log_every and ((loop_idx + 1) % cli_args.log_every == 0 or loop_idx == 0):
            print(f"[rank {rank}] loop {loop_idx + 1}/{cli_args.num_loops}: "
                  f"{throughput:.2f} samples/s ({elapsed:.2f}s)", flush=True)

    if len(local_ids) != total_samples_per_rank:
        raise RuntimeError(
            f"Rank {rank} expected {total_samples_per_rank} samples, received {len(local_ids)}")

    xm.rendezvous("collect_metrics")

    device = xm.xla_device()
    loop_tensor = torch.tensor(local_throughputs, dtype=torch.float32, device=device)
    id_tensor = torch.tensor(local_id_hashes, dtype=torch.int64, device=device)

    gathered_throughputs = xm.all_gather(loop_tensor, dim=0).cpu()
    gathered_ids = xm.all_gather(id_tensor, dim=0).cpu()

    global_loop_avg = gathered_throughputs.mean(dim=0).tolist()

    if xm.is_master_ordinal():
        per_rank_throughputs: Dict[int, List[float]] = {
            worker: gathered_throughputs[worker].tolist()
            for worker in range(world_size)
        }

        per_rank_id_hashes = gathered_ids.reshape(world_size, -1).numpy().tolist()
        flat_ids = [item for sublist in per_rank_id_hashes for item in sublist]
        expected_total = total_samples_per_rank * world_size
        if len(flat_ids) != expected_total:
            raise RuntimeError(
                f"Expected {expected_total} TFDS ids, gathered {len(flat_ids)}")

        unique_ids = set(flat_ids)
        duplicates = len(flat_ids) - len(unique_ids)
        if duplicates:
            raise RuntimeError(
                f"Detected {duplicates} duplicate TFDS ids across ranks; sharding is incorrect")

        print("✅ TFDS id uniqueness check passed", flush=True)

        for worker, series in sorted(per_rank_throughputs.items()):
            worker_max = max(series)
            worker_min = min(series)
            ratio = worker_min / max(worker_max, 1e-6)
            if ratio < cli_args.min_throughput_ratio:
                raise RuntimeError(
                    f"Rank {worker} throughput dropped below ratio threshold: {ratio:.2f} < "
                    f"{cli_args.min_throughput_ratio}")
            if cli_args.min_samples_per_sec > 0 and worker_min < cli_args.min_samples_per_sec:
                raise RuntimeError(
                    f"Rank {worker} min throughput {worker_min:.2f} < "
                    f"required {cli_args.min_samples_per_sec}")
            print(f"Rank {worker}: min {worker_min:.2f} / max {worker_max:.2f} samples/s "
                  f"(ratio {ratio:.2f})", flush=True)

        global_min = min(global_loop_avg)
        global_max = max(global_loop_avg)
        global_ratio = global_min / max(global_max, 1e-6)
        if global_ratio < cli_args.min_throughput_ratio:
            raise RuntimeError(
                f"Global throughput degraded below ratio threshold: {global_ratio:.2f} < "
                f"{cli_args.min_throughput_ratio}")

        print(f"Global throughput per loop (avg samples/s across ranks): "
              f"{', '.join(f'{v:.2f}' for v in global_loop_avg)}", flush=True)
        print("✅ Throughput stability check passed", flush=True)
        print("✅ Multihost TFDS loader stress test completed successfully", flush=True)


if __name__ == "__main__":
    main()
