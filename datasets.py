# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet data loading built on the official big_vision pipeline."""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "tensorflow and tensorflow_datasets must be installed to use the"
        " big_vision dataloader") from exc

from big_vision.pp import builder as pp_builder  # pylint: disable=import-error

# Ensure preprocessing ops are registered before building pipelines.
import big_vision.pp.ops_general  # pylint: disable=unused-import  # noqa: F401
import big_vision.pp.ops_image  # pylint: disable=unused-import  # noqa: F401
try:  # RandAug lives in archive.
    import big_vision.pp.archive.randaug  # pylint: disable=unused-import  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pass


@dataclasses.dataclass
class BigVisionLoaderConfig:
    """Configuration for the TFDS-based ImageNet loader."""

    name: str
    data_dir: Optional[str]
    split: str
    is_train: bool
    process_index: int
    process_count: int
    seed: int
    preprocess: str
    shuffle_buffer_size: int = 250_000
    cache: bool = False
    prefetch: int = 2
    num_parallel_calls: int = 100
    private_threadpool_size: int = 48
    normalize: str = "imagenet"
    skip_decode: bool = True
    return_tfds_id: bool = False


def _add_tpu_host_options(dataset: tf.data.Dataset,
                          private_threadpool_size: int) -> tf.data.Dataset:
    """Copy of big_vision's TPU host dataset options."""
    options = tf.data.Options()
    options.threading.private_threadpool_size = private_threadpool_size
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.inject_prefetch = False
    return dataset.with_options(options)


class BigVisionImageNetDataset(IterableDataset):
    """Iterable PyTorch dataset backed by the big_vision TFDS pipeline."""

    num_classes = 1000
    requires_distributed_sampler = False

    def __init__(self, config: BigVisionLoaderConfig):
        super().__init__()
        init_start = time.time()
        self._pid = os.getpid()
        self._log_prefix = (
            f"[BigVisionDataset][PID {self._pid}] split={config.split} "
            f"proc={config.process_index}/{config.process_count}"
        )
        print(
            f"{self._log_prefix} -> init start (train={config.is_train})",
            flush=True,
        )
        self._config = config
        if config.data_dir is None:
            raise ValueError(
                "TFDS data directory not provided. Set --tfds_data_dir or --data_path"
                " to the root containing imagenet2012 TFDS files.")
        builder_start = time.time()
        print(
            f"{self._log_prefix} -> constructing tfds builder at {config.data_dir}",
            flush=True,
        )
        self._builder = tfds.builder(
            config.name, data_dir=config.data_dir, try_gcs=True
        )
        print(
            f"{self._log_prefix} -> tfds builder ready in "
            f"{time.time() - builder_start:.2f}s",
            flush=True,
        )
        self._split = config.split
        base_split = self._split.split('[')[0]
        if base_split not in self._builder.info.splits:
            raise ValueError(f"Unsupported split '{self._split}'. Provide a"
                             " base TFDS split such as 'train' or 'validation'.")
        self._base_split = base_split
        self._global_size = self._builder.info.splits[base_split].num_examples
        self._local_size = self._compute_local_size(
            self._global_size, config.process_index, config.process_count
        )
        self._preprocess_fn = pp_builder.get_preprocess_fn(
            config.preprocess, log_data=False, log_steps=False)
        self._mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1).float()
        self._std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1).float()
        self._epoch = 0

        if config.process_count < 1:
            raise ValueError("process_count must be >= 1")
        if not (0 <= config.process_index < config.process_count):
            raise ValueError("process_index must satisfy 0 <= index < count")

        print(
            f"{self._log_prefix} -> init done in {time.time() - init_start:.2f}s "
            f"(global={self._global_size}, local={self._local_size})",
            flush=True,
        )

    @staticmethod
    def _compute_local_size(global_size: int, process_index: int,
                            process_count: int) -> int:
        per_proc = global_size // process_count
        remainder = global_size % process_count
        return per_proc + (1 if process_index < remainder else 0)

    def __len__(self) -> int:
        return self._global_size

    def _local_split(self) -> tfds.typing.ReadInstructionOrSplit:
        return tfds.even_splits(self._split, self._config.process_count)[
            self._config.process_index]

    def _build_tf_dataset(self, epoch_seed: Optional[int]) -> tf.data.Dataset:
        current_epoch = getattr(self, "_epoch", 0)
        should_log = current_epoch <= 3 or current_epoch % 5 == 0
        build_start = time.time()
        if should_log:
            print(
                f"{self._log_prefix} -> _build_tf_dataset start "
                f"(epoch={current_epoch}, seed={epoch_seed})",
                flush=True,
            )

        read_config = tfds.ReadConfig(
            skip_prefetch=True,
            try_autocache=False,
            add_tfds_id=True,
            shuffle_seed=epoch_seed,
        )
        decoders = None
        if self._config.skip_decode:
            decoders = {'image': tfds.decode.SkipDecoding()}

        stage_start = time.time()
        ds = self._builder.as_dataset(
            split=self._local_split(),
            shuffle_files=self._config.is_train,
            read_config=read_config,
            decoders=decoders,
        )
        if should_log:
            print(
                f"{self._log_prefix} -> as_dataset duration "
                f"{time.time() - stage_start:.2f}s",
                flush=True,
            )

        stage_start = time.time()
        ds = _add_tpu_host_options(ds, self._config.private_threadpool_size)
        if should_log:
            print(
                f"{self._log_prefix} -> add_host_options duration "
                f"{time.time() - stage_start:.2f}s",
                flush=True,
            )
        if self._config.cache:
            stage_start = time.time()
            ds = ds.cache()
            if should_log:
                print(
                    f"{self._log_prefix} -> cache() duration "
                    f"{time.time() - stage_start:.2f}s",
                    flush=True,
                )
        if self._config.is_train and self._config.shuffle_buffer_size > 0:
            stage_start = time.time()
            ds = ds.shuffle(
                self._config.shuffle_buffer_size,
                seed=epoch_seed,
                reshuffle_each_iteration=True,
            )
            if should_log:
                print(
                    f"{self._log_prefix} -> shuffle() duration "
                    f"{time.time() - stage_start:.2f}s",
                    flush=True,
                )
        stage_start = time.time()
        ds = ds.map(self._preprocess_fn,
                    num_parallel_calls=self._config.num_parallel_calls)
        if should_log:
            print(
                f"{self._log_prefix} -> map() duration {time.time() - stage_start:.2f}s",
                flush=True,
            )
        if self._config.prefetch:
            stage_start = time.time()
            ds = ds.prefetch(self._config.prefetch)
            if should_log:
                print(
                    f"{self._log_prefix} -> prefetch({self._config.prefetch}) duration "
                    f"{time.time() - stage_start:.2f}s",
                    flush=True,
                )
        if should_log:
            print(
                f"{self._log_prefix} -> _build_tf_dataset total "
                f"{time.time() - build_start:.2f}s",
                flush=True,
            )
        return ds

    def _maybe_normalize(self, image: torch.Tensor) -> torch.Tensor:
        if self._config.normalize == 'imagenet':
            return image.sub_(self._mean).div_(self._std)
        return image

    def __iter__(self) -> Iterator[
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, str]
    ]:
        self._epoch += 1
        epoch_seed = None
        if self._config.seed is not None:
            epoch_seed = self._config.seed + self._epoch

        should_log = self._epoch <= 3 or self._epoch % 5 == 0
        iter_start = time.time()
        if should_log:
            print(
                f"{self._log_prefix} -> iterator start epoch={self._epoch} "
                f"seed={epoch_seed}",
                flush=True,
            )

        ds = self._build_tf_dataset(epoch_seed)
        if should_log:
            print(
                f"{self._log_prefix} -> tf.data pipeline built in "
                f"{time.time() - iter_start:.2f}s",
                flush=True,
            )

        iterator_start = time.time()
        dataset_iterator = ds.as_numpy_iterator()
        if should_log:
            print(
                f"{self._log_prefix} -> numpy iterator ready in "
                f"{time.time() - iterator_start:.2f}s",
                flush=True,
            )

        example_count = 0
        fetch_start = time.time()
        for example in dataset_iterator:
            fetch_elapsed = time.time() - fetch_start
            example_count += 1
            if should_log and (example_count <= 5 or example_count % 10000 == 0):
                print(
                    f"{self._log_prefix} -> fetched example {example_count} in "
                    f"{fetch_elapsed:.3f}s",
                    flush=True,
                )

            image = example['image']
            label = int(example['label'])
            tfds_id = example.get('tfds_id')
            if tfds_id is not None:
                if isinstance(tfds_id, bytes):
                    tfds_id = tfds_id.decode('utf-8')
                else:
                    tfds_id = str(tfds_id)

            if image.dtype != np.float32:
                image = image.astype(np.float32)
            # tf.data returns images in HWC; convert to CHW for PyTorch.
            image = np.transpose(image, (2, 0, 1))
            torch_image = torch.from_numpy(image).contiguous()
            torch_image = self._maybe_normalize(torch_image)
            label_tensor = torch.tensor(label, dtype=torch.long)

            if self._config.return_tfds_id and tfds_id is not None:
                yield torch_image, label_tensor, tfds_id
            else:
                yield torch_image, label_tensor

            fetch_start = time.time()

        if should_log:
            print(
                f"{self._log_prefix} -> iterator exhausted after {example_count} "
                f"examples in {time.time() - iter_start:.2f}s",
                flush=True,
            )


def build_dataset(is_train: bool, args):
    """Builds datasets using the big_vision TFDS input pipeline."""

    if args.data_set not in {'IMNET', 'imagenet2012'}:
        raise NotImplementedError(
            "The rewritten dataloader currently supports ImageNet via TFDS."
            " Requested data_set='%s'" % args.data_set)

    data_dir = getattr(args, 'tfds_data_dir', None) or (
        args.data_path if is_train or not getattr(args, 'eval_data_path', None)
        else args.eval_data_path)

    process_count = max(1, getattr(args, 'world_size', 1))
    process_index = max(0, getattr(args, 'rank', 0))

    split = args.tfds_train_split if is_train else args.tfds_eval_split
    preprocess = (args.big_vision_pp_train if is_train
                  else args.big_vision_pp_eval)
    cache = args.tfds_cache_raw if is_train else args.tfds_cache_eval

    shuffle_buffer = getattr(args, 'tfds_shuffle_buffer', 250_000)
    if not is_train:
        shuffle_buffer = 0

    build_start = time.time()
    print(
        f"[build_dataset] start is_train={is_train} split={split} "
        f"rank={process_index}/{process_count} data_dir={data_dir}",
        flush=True,
    )

    config = BigVisionLoaderConfig(
        name='imagenet2012',
        data_dir=data_dir,
        split=split,
        is_train=is_train,
        process_index=process_index,
        process_count=process_count,
        seed=getattr(args, 'seed', 0),
        preprocess=preprocess,
        shuffle_buffer_size=shuffle_buffer,
        cache=cache,
        prefetch=getattr(args, 'tfds_prefetch', 2),
        num_parallel_calls=getattr(args, 'tfds_num_parallel_calls', 100),
        private_threadpool_size=getattr(args, 'tfds_private_threadpool_size', 48),
        normalize=getattr(args, 'big_vision_normalize', 'imagenet'),
        skip_decode=getattr(args, 'tfds_skip_decode', True),
        return_tfds_id=getattr(args, 'tfds_return_id', False),
    )

    dataset = BigVisionImageNetDataset(config)
    print(
        f"[build_dataset] done in {time.time() - build_start:.2f}s "
        f"(global={len(dataset)}, local={dataset._local_size})",
        flush=True,
    )
    return dataset, dataset.num_classes
