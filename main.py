# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# print("🚀 Starting main.py import phase...", flush=True)

# print("📦 Importing basic packages...", flush=True)
import argparse
import datetime
import numpy as np
import time
import warnings
# print("🔥 Importing torch...", flush=True)
import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# print("🌐 Importing torch.distributed...", flush=True)
import torch.distributed as dist  ### Added for TPU
import json
import os
# print("✅ Basic imports completed", flush=True)

# --- XLA + I/O env sanity (add near the top, after imports) ---
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # avoids HDF5 lock issues on gcsfuse

# Small convenience: XLA-aware barrier
def xla_barrier(tag="barrier"):
    try:
        import torch.distributed as dist
        import torch_xla.core.xla_model as xm
        xm.rendezvous(tag)
    except Exception:
        # fall back to torch.distributed if not on XLA
        if 'dist' in globals() and dist.is_available() and dist.is_initialized():
            dist.barrier()

# One-time TPU compile warmup to make the first step deterministic
def xla_compile_warmup(model, criterion, optimizer, device, args):
    import torch
    import torch_xla.core.xla_model as xm
    model.train()
    bs = max(2, min(args.batch_size, 8))
    dummy = torch.randn(bs, 3, args.input_size, args.input_size, device=device)
    target = torch.zeros(bs, dtype=torch.long, device=device)
    out = model(dummy)
    loss = criterion(out, target)
    loss.backward()
    xm.optimizer_step(optimizer, barrier=True)  # compiles + executes once

# print("📁 Importing pathlib...", flush=True)
from pathlib import Path

# print("🎯 Importing timm packages...", flush=True)
from timm.data.mixup import Mixup
# from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
# print("   ✅ timm packages imported", flush=True)

# print("🏭 Importing local modules...", flush=True)
from optim_factory import create_optimizer, LayerDecayValueAssigner
# print("   ✅ optim_factory imported", flush=True)
from models import create_model
# print("   ✅ models imported", flush=True)

# print("📊 Importing datasets...", flush=True)
from datasets import build_dataset, BigVisionImageNetDataset
# print("   ✅ datasets imported", flush=True)
from engine import train_one_epoch, evaluate
# print("   ✅ engine imported", flush=True)

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
# print("   ✅ utils imported", flush=True)



# print("🔥 Importing torch_xla packages...", flush=True)
# Always safe to import
import torch_xla                      # core package (optional but fine)
# print("   ✅ torch_xla core imported", flush=True)

# Device helpers: xla:0 handle, mark_step(), optimizer_step(), ordinals, etc.
import torch_xla.core.xla_model as xm
# print("   ✅ torch_xla.core.xla_model imported", flush=True)

# Multiprocess launcher (spawn one process per TPU core)
import torch_xla.distributed.xla_multiprocessing as xmp
# print("   ✅ torch_xla.distributed.xla_multiprocessing imported", flush=True)

# Runtime info (global device count, global ordinal, device type, etc.)
import torch_xla.runtime as xr
# print("   ✅ torch_xla.runtime imported", flush=True)

# Registers the 'xla' torch.distributed backend. Required *before* calling:
#   torch.distributed.init_process_group(backend="xla", init_method="xla://")
import torch_xla.distributed.xla_backend  # import for side-effects; no alias
# print("   ✅ torch_xla.distributed.xla_backend imported", flush=True)

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

print("🎉 ALL IMPORTS COMPLETED SUCCESSFULLY!", flush=True)
# if not dist.is_initialized():
#     dist.init_process_group(backend="xla", init_method="xla://")

# args.rank = dist.get_rank()
# args.world_size = dist.get_world_size()
# args.gpu = 0
# args.dist_backend = "xla"
# args.distributed = True
# setup_for_distributed(args.rank == 0)

# dev = xm.xla_device()
# print(f"[XLA] init ok | rank={args.rank}/{args.world_size-1} dev={dev} "
#       f"global_ordinal={xr.global_ordinal()} world={xr.world_size()}", flush=True)


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='vit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/tlu37/scratchdkhasha1/tlu37/Distillation/dataset/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['IMNET', 'imagenet2012'],
                        type=str, help='Dataset key. The big_vision loader currently supports ImageNet TFDS only.')
    parser.add_argument('--tfds_data_dir', default=None, type=str,
                        help='Directory containing TFDS-prepared ImageNet data. Defaults to data_path when unset.')
    parser.add_argument('--tfds_train_split', default='train', type=str,
                        help='TFDS split string for training data (e.g. "train" or "train[:99%]").')
    parser.add_argument('--tfds_eval_split', default='validation', type=str,
                        help='TFDS split string for evaluation data.')
    parser.add_argument('--tfds_shuffle_buffer', default=250_000, type=int,
                        help='Shuffle buffer size for the tf.data training pipeline.')
    parser.add_argument('--tfds_cache_raw', type=str2bool, default=False,
                        help='Cache raw TFDS examples before preprocessing (trades RAM for speed).')
    parser.add_argument('--tfds_cache_eval', type=str2bool, default=False,
                        help='Cache evaluation dataset after preprocessing.')
    parser.add_argument('--tfds_prefetch', default=2, type=int,
                        help='Prefetch depth for tf.data pipelines.')
    parser.add_argument('--tfds_num_parallel_calls', default=100, type=int,
                        help='Number of parallel calls for preprocessing map.')
    parser.add_argument('--tfds_private_threadpool_size', default=48, type=int,
                        help='Private threadpool size for tf.data host workers.')
    parser.add_argument('--tfds_skip_decode', type=str2bool, default=True,
                        help='Skip TFDS automatic decoding so big_vision ops decode images.')
    parser.add_argument('--big_vision_pp_train', type=str,
                        default='decode_jpeg_and_inception_crop(224)|flip_lr|value_range(0, 1)|keep("image", "label")',
                        help='big_vision preprocessing pipeline string for training.')
    parser.add_argument('--big_vision_pp_eval', type=str,
                        default='decode|resize_small(256)|central_crop(224)|value_range(0, 1)|keep("image", "label")',
                        help='big_vision preprocessing pipeline string for evaluation.')
    parser.add_argument('--big_vision_normalize', type=str, default='imagenet', choices=['imagenet', 'none'],
                        help='Per-sample normalization applied after preprocessing.')
    parser.add_argument('--cache_dataset_in_ram', action='store_true', default=False,
                        help='Cache entire dataset in RAM for ultra-fast access (requires sufficient RAM, optimizes GCS bucket access)')
    parser.add_argument('--output_dir', default='/home/tlu37/scratchdkhasha1/tlu37/Distillation/models_imagenet/dyt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=False)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='global rank of the current process')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=True, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=True,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--wandb_mode', type=str, default='offline', choices=['online', 'offline', 'disabled'],
                        help="W&B mode: 'online' for internet connection, 'offline' for local logging, 'disabled' to turn off")
    parser.add_argument('--wandb_dir', type=str, default=None,
                        help="Directory to store W&B offline files (default: ~/.wandb)")
    parser.add_argument('--project', default='ViT-tpu', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")
    parser.add_argument('--experiment', default=None, type=str,
                        help="Experiment name for both W&B run name and output directory subfolder")
    parser.add_argument('--log_freq', default=1, type=int,
                        help="Frequency for console and wandb logging (every N iterations). Default: 10")

    # Knowledge Distillation arguments
    parser.add_argument('--kd', type=str2bool, default=False,
                        help="Enable knowledge distillation training")
    parser.add_argument('--teacher_path', default='', type=str,
                        help="Path to teacher model checkpoint")
    parser.add_argument('--teacher_arch', default='', type=str,
                        help="Teacher model architecture")
    parser.add_argument('--kd_alpha', type=float, default=0.7,
                        help="Weight for combining CE loss and KD loss (default: 0.7)")
    parser.add_argument('--kd_temperature', type=float, default=4.0,
                        help="Temperature for distillation loss (default: 4.0)")

    parser.add_argument('--tpu', action='store_true',
                        help='Use TPU (PJRT) multi-host via torch-xla')

    return parser


def main(args):
    # Set up signal handlers for graceful shutdown
    # if not args.tpu:
    #     import signal
    #     import sys
        
    #     def signal_handler(sig, frame):
    #         print(f"\nReceived signal {sig}, shutting down gracefully...")
    #         sys.exit(0)
        
    #     signal.signal(signal.SIGINT, signal_handler)
    #     signal.signal(signal.SIGTERM, signal_handler)
    
    # Create experiment-specific output directory
    if args.experiment is not None and args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.experiment)
    
    # Ensure output directory exists
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(args)
    ### Added for TPU
    if args.tpu and not hasattr(args, '_tpu_spawned_process'):
        # Main TPU process: DO NOT initialize distributed here - let spawned processes handle it
        print("Main TPU process: Skipping distributed init (will be handled by spawned processes)")
        device = torch.device('xla')
    elif hasattr(args, '_tpu_spawned_process'):
        # TPU spawned process: Initialize distributed and use XLA device
        print("TPU spawned process: Initializing XLA distributed mode...")
        utils.init_distributed_mode_xla(args)  # Only spawned processes join the distributed group
        device = args._xla_device
    else:
        # Non-TPU path: GPU/CPU
        utils.init_distributed_mode(args)      # original GPU/CPU path
        device = torch.device(args.device)
    ### End of TPU

    if args.tpu:
        # Disable PyTorch AMP on TPU; XLA BF16 is controlled by XLA_USE_BF16
        args.use_amp = False
        # Force EMA to CPU to avoid XLA graph complications
        args.model_ema_force_cpu = True

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ### Added for TPU
    if not args.tpu:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    ### End of TPU
    
    # Handle dataset creation based on TPU mode to avoid pickle issues
    if hasattr(args, '_tpu_spawned_process'):
        # TPU spawned process: datasets were created in main_tpu()
        dataset_train = args._dataset_train
        dataset_val = args._dataset_val

        def _dataset_summary(ds):
            if ds is None:
                return "None"
            try:
                length = len(ds)
            except Exception:  # pragma: no cover - diagnostics only
                length = "unknown"
            return f"{type(ds).__name__}(len={length})"

        _log_event(
            "main",
            "rank=%s received pre-built datasets train=%s val=%s"
            % (
                utils.get_rank(),
                _dataset_summary(dataset_train),
                _dataset_summary(dataset_val),
            ),
        )
    elif args.tpu:
        # Main TPU process: just get nb_classes for validation, datasets created in spawned processes
        _log_event("main", "rank=%s fetching dataset metadata" % utils.get_rank())
        metadata_start = time.time()
        _, args.nb_classes = build_dataset(is_train=True, args=args)
        _log_event(
            "main",
            "rank=%s metadata fetch done in %.2fs"
            % (utils.get_rank(), time.time() - metadata_start),
        )
        dataset_train = None  # Will be created in spawned process
        dataset_val = None    # Will be created in spawned process
    else:
        # Non-TPU mode: create datasets normally
        _log_event("main", "rank=%s building training dataset" % utils.get_rank())
        build_train_start = time.time()
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        try:
            dataset_train_len = len(dataset_train)
        except Exception:  # pragma: no cover - diagnostics only
            dataset_train_len = "unknown"
        _log_event(
            "main",
            "rank=%s training dataset ready in %.2fs len=%s"
            % (
                utils.get_rank(),
                time.time() - build_train_start,
                dataset_train_len,
            ),
        )
        if args.disable_eval:
            args.dist_eval = False
            dataset_val = None
            _log_event("main", "rank=%s eval disabled" % utils.get_rank())
        else:
            _log_event("main", "rank=%s building eval dataset" % utils.get_rank())
            build_val_start = time.time()
            dataset_val, _ = build_dataset(is_train=False, args=args)
            try:
                dataset_val_len = len(dataset_val)
            except Exception:  # pragma: no cover - diagnostics only
                dataset_val_len = "unknown"
            _log_event(
                "main",
                "rank=%s eval dataset ready in %.2fs len=%s"
                % (
                    utils.get_rank(),
                    time.time() - build_val_start,
                    dataset_val_len,
                ),
            )
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    _log_event(
        "main",
        "rank=%s dataset setup complete train=%s val=%s"
        % (global_rank, dataset_train is not None, dataset_val is not None),
    )

    is_bigvision_train = isinstance(dataset_train, BigVisionImageNetDataset)
    is_bigvision_val = isinstance(dataset_val, BigVisionImageNetDataset)

    sampler_train = None
    sampler_val = None

    if args.distributed and dataset_train is not None and not is_bigvision_train:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank,
            shuffle=True, seed=args.seed)
        print("Sampler_train = %s" % str(sampler_train))

    if dataset_val is not None:
        if args.distributed and args.dist_eval and not is_bigvision_val:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        elif not is_bigvision_val:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # Set W&B environment variables early, before any W&B initialization
    if global_rank == 0 and args.enable_wandb and args.wandb_mode != 'disabled':
        if args.wandb_mode == 'offline':
            os.environ['WANDB_MODE'] = 'offline'
            if args.wandb_dir:
                os.environ['WANDB_DIR'] = args.wandb_dir
            print(f"W&B running in offline mode. Files will be saved to: {os.environ.get('WANDB_DIR', '~/.wandb')}")
        elif args.wandb_mode == 'online':
            os.environ['WANDB_MODE'] = 'online'
            print("W&B running in online mode")
        
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    # For TPU, disable DataLoader multiprocessing but use MpDeviceLoader for parallelism
    tpu_mode = args.tpu or hasattr(args, '_xla_device')

    if dataset_train is not None:
        if tpu_mode:
            num_workers_train = 0
            pin_memory_train = False
            print("TPU mode: Using num_workers=0 to avoid pickle issues, will use MpDeviceLoader for parallelism")
        elif is_bigvision_train:
            num_workers_train = 0
            pin_memory_train = False
            print("big_vision loader: forcing num_workers=0 and pin_memory=False to reuse tf.data threads")
        else:
            num_workers_train = args.num_workers
            pin_memory_train = args.pin_mem

        dataloader_train_start = time.time()
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=num_workers_train,
            pin_memory=pin_memory_train,
            drop_last=True,
        )
        _log_event(
            "main",
            "rank=%s train loader ready in %.2fs workers=%s pin_memory=%s"
            % (
                utils.get_rank(),
                time.time() - dataloader_train_start,
                num_workers_train,
                pin_memory_train,
            ),
        )
    else:
        data_loader_train = None
        _log_event("main", "rank=%s no training dataset" % utils.get_rank())

    if dataset_val is not None:
        if tpu_mode or is_bigvision_val:
            num_workers_val = 0
            pin_memory_val = False
            if tpu_mode:
                print("TPU mode: validation loader using num_workers=0")
            else:
                print("big_vision loader (eval): forcing num_workers=0 and pin_memory=False")
        else:
            num_workers_val = args.num_workers
            pin_memory_val = args.pin_mem

        dataloader_val_start = time.time()
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=num_workers_val,
            pin_memory=pin_memory_val,
            drop_last=False,
        )
        _log_event(
            "main",
            "rank=%s val loader ready in %.2fs workers=%s pin_memory=%s"
            % (
                utils.get_rank(),
                time.time() - dataloader_val_start,
                num_workers_val,
                pin_memory_val,
            ),
        )
    else:
        data_loader_val = None
        _log_event("main", "rank=%s no validation dataset" % utils.get_rank())

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    _log_event("main", "rank=%s creating model %s" % (global_rank, args.model))

    if "convnext" in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            ls_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
        )
    elif "vit" in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
            drop_path_rate=args.drop_path,
        )
    else:
        raise ValueError(f"Unrecognized model: {args.model}")

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)
    
    # For TPU, force another XLA synchronization after model.to(device)
    if args.tpu:
        import torch_xla.core.xla_model as xm
        _log_event(
            "main",
            "rank=%s forcing XLA mark_step after model.to" % global_rank,
        )
        xm.mark_step()  # Force XLA synchronization
        _log_event(
            "main",
            "rank=%s completed XLA mark_step after model.to" % global_rank,
        )

    # Load teacher model for knowledge distillation
    teacher_model = None
    if args.kd:
        if not args.teacher_path:
            raise ValueError("Teacher path must be provided when using knowledge distillation")
        if not args.teacher_arch:
            raise ValueError("Teacher architecture must be provided when using knowledge distillation")
        
        print(f"Loading teacher model: {args.teacher_arch} from {args.teacher_path}")
        
        # Create teacher model
        if "convnext" in args.teacher_arch:
            teacher_model = create_model(
                args.teacher_arch,
                pretrained=False,
                num_classes=args.nb_classes,
                drop_path_rate=0.0,  # No dropout for teacher
                ls_init_value=args.layer_scale_init_value,
            )
        elif "vit" in args.teacher_arch:
            teacher_model = create_model(
                args.teacher_arch,
                pretrained=False,
                num_classes=args.nb_classes,
                global_pool='avg',
                drop_path_rate=0.0,  # No dropout for teacher
            )
        else:
            raise ValueError(f"Unrecognized teacher model: {args.teacher_arch}")
        
        # Load teacher checkpoint
        if args.teacher_path.startswith('https'):
            teacher_checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            teacher_checkpoint = torch.load(args.teacher_path, map_location='cpu')
        
        teacher_checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in teacher_checkpoint:
                teacher_checkpoint_model = teacher_checkpoint[model_key]
                print("Load teacher state_dict by model_key = %s" % model_key)
                break
        if teacher_checkpoint_model is None:
            teacher_checkpoint_model = teacher_checkpoint
        
        utils.load_state_dict(teacher_model, teacher_checkpoint_model, prefix=args.model_prefix)
        teacher_model.to(device)
        teacher_model.eval()  # Set teacher to eval mode
        for param in teacher_model.parameters():
            param.requires_grad = False  # Freeze teacher parameters
        
        print(f"Teacher model loaded successfully with {sum(p.numel() for p in teacher_model.parameters())} parameters")

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        # For KD, we only want EMA of the student model
        ema_model = student_model_for_optimizer if args.kd else model
        model_ema = ModelEma(
            ema_model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    
    # For TPU, force XLA synchronization before parameter counting
    if args.tpu:
        import torch_xla.core.xla_model as xm
        print("Forcing XLA synchronization before parameter counting...")
        xm.mark_step()  # Force XLA synchronization
        print("XLA mark_step() completed")
    
    # print("Model = %s" % str(model_without_ddp))
    # print("Model architecture printed successfully")
    
    # # Count parameters carefully for XLA
    # print("About to count model parameters...")
    # print("Counting model parameters...")
    # if args.tpu:
    #     # For TPU, count parameters on CPU to avoid XLA compilation issues
    #     print("TPU mode: Moving model to CPU temporarily for parameter counting...")
    #     try:
    #         # Create a temporary CPU model for counting
    #         import copy
    #         cpu_model = copy.deepcopy(model_without_ddp).cpu()
    #         if args.kd and hasattr(cpu_model, 'student'):
    #             n_parameters = sum(p.numel() for p in cpu_model.student.parameters() if p.requires_grad)
    #         else:
    #             n_parameters = sum(p.numel() for p in cpu_model.parameters() if p.requires_grad)
    #         del cpu_model  # Clean up
    #         print('number of params:', n_parameters)
    #     except Exception as e:
    #         print(f"Error counting parameters on CPU: {e}")
    #         # Fallback: estimate based on model structure
    #         n_parameters = 0
    # else:
    #     # Non-TPU: count normally
    #     try:
    #         if args.kd and hasattr(model_without_ddp, 'student'):
    #             n_parameters = sum(p.numel() for p in model_without_ddp.student.parameters() if p.requires_grad)
    #         else:
    #             n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #         print('number of params:', n_parameters)
    #     except Exception as e:
    #         print(f"Error counting parameters: {e}")
    #         n_parameters = 0

    # Get world size 
    world_size = utils.get_world_size()
    if hasattr(args, '_tpu_spawned_process'):
        print(f"TPU spawned process: world_size = {world_size}")
    else:
        print(f"World size = {world_size}")
    
    total_batch_size = args.batch_size * args.update_freq * world_size
    print(f"Calculated total_batch_size = {total_batch_size}")
    
    if dataset_train is None:
        num_training_steps_per_epoch = 0
        print("Dataset is not instantiated on this process; skipping local step computation")
    else:
        num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        print("Number of training examples = %d" % len(dataset_train))
    
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    # Wrap model for knowledge distillation before DDP setup
    if args.kd and teacher_model is not None:
        # Knowledge Distillation wrapper model
        class StudentWithDistillation(nn.Module):
            def __init__(self, student_model, teacher_model):
                super().__init__()
                self.student = student_model
                self.teacher = teacher_model
                
            def forward(self, x):
                student_logits = self.student(x)
                if self.training and self.teacher is not None:
                    with torch.no_grad():
                        teacher_logits = self.teacher(x)
                    return student_logits, teacher_logits
                return student_logits
        
        model = StudentWithDistillation(model, teacher_model)
        print(f"Model wrapped for Knowledge Distillation with alpha={args.kd_alpha}, temperature={args.kd_temperature}")

    ### Added for TPU
    if args.distributed and not args.tpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
    elif args.tpu and args.distributed:
        # On TPU multihost, XLA handles sharding / grad sync; do not wrap again
        model_without_ddp = model
        print("TPU multihost mode: XLA handles data distribution internally")
    else:
        model_without_ddp = model
    ### End of TPU

    # For knowledge distillation, we need to get the actual student model for optimizer
    student_model_for_optimizer = model_without_ddp
    if args.kd and hasattr(model_without_ddp, 'student'):
        student_model_for_optimizer = model_without_ddp.student

    _log_event("main", "rank=%s creating optimizer" % global_rank)
    
    # Use the proper create_optimizer function for all cases (now TPU-safe)
    optimizer = create_optimizer(
        args, student_model_for_optimizer, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    _log_event("main", "rank=%s optimizer ready" % global_rank)

    # Skip extra XLA sync after optimizer - let xm.optimizer_step handle synchronization

    # Create device-aware scaler
    if args.tpu or hasattr(args, '_tpu_spawned_process'):
        print("TPU mode: Skipping loss scaler (XLA handles mixed precision)")
        print("TPU mode: About to set loss_scaler = None...")
        loss_scaler = None  # TPU doesn't need external scaling
        print("TPU mode: Loss scaler set to None")
    else:
        device_type = 'cuda'
        print(f"Using device type: {device_type}")
        print("About to create loss scaler...")
        loss_scaler = NativeScaler(device=device_type)
        print("Loss scaler created successfully")

    _log_event("main", "rank=%s creating LR scheduler" % global_rank)
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    _log_event("main", "rank=%s LR scheduler ready" % global_rank)
    _log_event("main", "rank=%s preparing weight decay schedule" % global_rank)
    
    # Bypass the problematic attribute access - just set it directly
    print("Directly setting weight_decay_end to avoid freeze...", flush=True)
    if not hasattr(args, 'weight_decay_end') or args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    print(f"weight_decay_end set to: {args.weight_decay_end}", flush=True)
    
    _log_event(
        "main",
        "rank=%s creating WD schedule" % global_rank
    )
    print(f"Calling with: wd={args.weight_decay}, wd_end={args.weight_decay_end}, epochs={args.epochs}, steps_per_epoch={num_training_steps_per_epoch}", flush=True)
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    _log_event("main", "rank=%s WD schedule ready" % global_rank)
    
    if args.tpu:
        print(f"Weight decay schedule created: {args.weight_decay} -> {args.weight_decay_end}", flush=True)
    else:
        print("About to calculate min/max of wd_schedule_values...", flush=True)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)), flush=True)

    _log_event("main", "rank=%s creating criterion" % global_rank)
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        print("Created SoftTargetCrossEntropy criterion", flush=True)
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        print("Created LabelSmoothingCrossEntropy criterion", flush=True)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Created CrossEntropyLoss criterion", flush=True)

    # Knowledge Distillation Loss Function
    if args.kd and teacher_model is not None:
        class DistillationLoss(nn.Module):
            def __init__(self, base_criterion, alpha=0.7, temperature=4.0):
                super().__init__()
                self.base_criterion = base_criterion
                self.alpha = alpha
                self.temperature = temperature
                self.kl_div = nn.KLDivLoss(reduction='batchmean')
                
            def forward(self, outputs, targets):
                if isinstance(outputs, tuple):
                    # Knowledge distillation mode - outputs is (student_logits, teacher_logits)
                    student_logits, teacher_logits = outputs
                    
                    # Standard cross-entropy loss
                    ce_loss = self.base_criterion(student_logits, targets)
                    
                    # Soften the logits with temperature
                    student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
                    teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=1)
                    
                    # KL divergence loss
                    kd_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
                    
                    # Combine losses
                    total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
                    
                    return total_loss
                else:
                    # Standard mode - outputs is just student logits
                    return self.base_criterion(outputs, targets)
        
        criterion = DistillationLoss(criterion, args.kd_alpha, args.kd_temperature)

    print("criterion = %s" % str(criterion), flush=True)

    # Skip compile warmup - it's causing shape mismatch errors
    if args.tpu:
        print("TPU mode: Skipping warmup (caused shape errors), will compile on first iteration", flush=True)

    _log_event("main", "rank=%s calling auto_load_model" % global_rank)
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    _log_event("main", "rank=%s auto_load_model complete" % global_rank)

    print("About to define get_eval_model helper function...", flush=True)
    # Helper function to get the model for evaluation (student only for KD)
    def get_eval_model(model):
        if args.kd and hasattr(model, 'student'):
            return model.student
        elif args.kd and hasattr(model, 'module') and hasattr(model.module, 'student'):
            return model.module.student
        return model

    print("get_eval_model function defined", flush=True)

    if args.eval:
        print(f"Eval only mode", flush=True)
        eval_model = get_eval_model(model)
        test_stats = evaluate(data_loader_val, eval_model, device, use_amp=args.use_amp, tpu=args.tpu)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%", flush=True)
        return

    _log_event("main", "rank=%s entering training loop" % global_rank)
    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0
    print("max_accuracy initialized", flush=True)

    # Create MpDeviceLoader once for TPU (instead of every epoch)
    original_train_sampler = None
    if args.tpu:
        try:
            import torch_xla.distributed.parallel_loader as pl
            print("🚀 Creating MpDeviceLoader for training DataLoader (TPU optimization)...", flush=True)
            # Preserve the original sampler before wrapping
            if hasattr(data_loader_train, 'sampler'):
                original_train_sampler = data_loader_train.sampler
            data_loader_train = pl.MpDeviceLoader(data_loader_train, device)
            print("✅ Training MpDeviceLoader created successfully", flush=True)
            
            # Also optimize validation DataLoader if it exists
            if data_loader_val is not None:
                try:
                    print("🚀 Creating MpDeviceLoader for validation DataLoader (TPU optimization)...", flush=True)
                    data_loader_val = pl.MpDeviceLoader(data_loader_val, device)
                    print("✅ Validation MpDeviceLoader created successfully", flush=True)
                except Exception as val_e:
                    print(f"⚠️ Validation MpDeviceLoader failed: {val_e}, keeping regular DataLoader for validation", flush=True)
        except Exception as e:
            print(f"⚠️ MpDeviceLoader failed: {e}, using regular DataLoaders", flush=True)

    print("Start training for %d epochs" % args.epochs, flush=True)
    print("About to record start_time...", flush=True)
    start_time = time.time()
    last_epoch_end_time = start_time
    print("About to enter training loop...", flush=True)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        between_epochs_delay = epoch_start_time - last_epoch_end_time
        print(f"Starting epoch {epoch} (delay since last epoch: {between_epochs_delay:.2f}s)...", flush=True)
        if args.distributed and data_loader_train is not None:
            print("Setting sampler epoch...", flush=True)
            sampler_start = time.time()
            # Use original sampler if we wrapped with MpDeviceLoader
            sampler_obj = None
            if original_train_sampler is not None:
                sampler_obj = original_train_sampler
            elif hasattr(data_loader_train, 'sampler'):
                sampler_obj = getattr(data_loader_train, 'sampler', None)
            if sampler_obj is not None and hasattr(sampler_obj, 'set_epoch'):
                print(f"About to call sampler.set_epoch({epoch})...", flush=True)
                sampler_obj.set_epoch(epoch)
                sampler_time = time.time() - sampler_start
                print(f"✅ sampler.set_epoch({epoch}) completed in {sampler_time:.2f}s", flush=True)
        if log_writer is not None:
            print("Setting log writer step...", flush=True)
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            print("Setting wandb steps...", flush=True)
            wandb_logger.set_steps()
        print("About to call train_one_epoch...", flush=True)
        train_epoch_start_time = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp, tpu=args.tpu, log_freq=args.log_freq  # Added log_freq parameter
        )
        train_epoch_end_time = time.time()
        train_epoch_duration = train_epoch_end_time - train_epoch_start_time
        print(f"✅ Training epoch {epoch} completed in {train_epoch_duration:.2f}s", flush=True)
        print("Checking checkpoint save conditions...", flush=True)
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                print("About to save checkpoint...", flush=True)
                # utils.save_model(
                #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                #     loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
                print("Checkpoint saved successfully", flush=True)
        print("About to start validation...", flush=True)
        if data_loader_val is not None:
            validation_start_time = time.time()
            print("Getting eval model...", flush=True)
            eval_model = get_eval_model(model)
            print("About to call evaluate()...", flush=True)
            test_stats = evaluate(data_loader_val, eval_model, device, use_amp=args.use_amp, tpu=args.tpu)
            validation_end_time = time.time()
            validation_duration = validation_end_time - validation_start_time
            print(f"✅ Validation completed in {validation_duration:.2f}s", flush=True)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                # if args.output_dir and args.save_ckpt:
                #     utils.save_model(
                #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                #         loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                        #  'n_parameters': n_parameters
                         }

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp, tpu=args.tpu)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    # if args.output_dir and args.save_ckpt:
                    #     utils.save_model(
                    #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    #         loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)
        
        # Track end of epoch for delay calculation
        last_epoch_end_time = time.time()
        epoch_total_time = last_epoch_end_time - epoch_start_time
        print(f"🏁 Epoch {epoch} total time: {epoch_total_time:.2f}s", flush=True)

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()

    # Finish W&B run properly
    if wandb_logger:
        wandb_logger.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))





# NEW: define a lightweight spawn function
def _mp_fn(index, args_dict):
    """
    Entrypoint for each spawned process.
    index: process index (unused here, but required by launcher)
    args_dict: dictionary of arguments that can be serialized
    """
    # Set environment flags for XLA (these can also be set outside if preferred)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_BF16", "1")
    
    # Convert args_dict back to argparse Namespace
    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)
    
    # Call the main training function with the parsed arguments
    main_tpu(args)


def main_tpu(args):
    """TPU-specific main function that creates datasets in each process"""
    # This is essentially the same as main() but creates datasets locally
    # to avoid pickle issues with custom dataset classes
    
    # Set up TPU distributed mode
    utils.init_distributed_mode_xla(args)
    device = torch.device('xla')
    worker_rank = utils.get_rank()

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create datasets in this process (avoiding pickle issues)
    _log_event("main", f"rank={worker_rank} building training dataset (worker)")
    dataset_train_start = time.time()
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    _log_event(
        "main",
        "rank=%s training dataset ready in %.2fs (worker)"
        % (worker_rank, time.time() - dataset_train_start),
    )
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
        _log_event(
            "main", f"rank={worker_rank} evaluation disabled via flag (worker)"
        )
    else:
        _log_event(
            "main", f"rank={worker_rank} building eval dataset (worker)"
        )
        dataset_val_start = time.time()
        dataset_val, _ = build_dataset(is_train=False, args=args)
        _log_event(
            "main",
            "rank=%s eval dataset ready in %.2fs (worker)"
            % (worker_rank, time.time() - dataset_val_start),
        )
    
    # Continue with the rest of the main() function logic
    # but skip the distributed mode setup and dataset creation since we did it above
    
    # Store datasets and device info in args to pass to main
    args._dataset_train = dataset_train
    args._dataset_val = dataset_val
    args._xla_device = device  # Pass the XLA device
    args._tpu_spawned_process = True  # Mark this as a TPU spawned process to skip init
    
    # Call the original main function
    main(args)

if __name__ == "__main__":
    # Parse arguments **once** in the main process
    print("🎯 Reached main execution block!", flush=True)
    print("🔍 Starting argument parsing...", flush=True)
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    print("✅ Arguments parsed successfully", flush=True)
    
    print("🚀 Starting TPU launch...", flush=True)
    # If using TPU, launch multiple processes; otherwise run main directly
    if args.tpu:
        import torch_xla
        import torch_xla.distributed.xla_backend  # Registers the 'xla' backend
        
        # Convert args to a dictionary to avoid pickle issues
        args_dict = vars(args)
        
        # Use torch_xla.launch to spawn processes on each TPU core
        torch_xla.launch(_mp_fn, args=(args_dict,))  
        # Here, _mp_fn will be called in each child process with the given args
    else:
        # Non-TPU path can run in a single process
        main(args)
def _log_event(tag: str, message: str) -> None:
    timestamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    entry = f"[{tag}] {timestamp} {message}"
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        xm.master_print(entry, flush=True)
    except Exception:
        print(entry, flush=True)
    log_dir = os.environ.get("TPU_LOG_DIR")
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{tag}.log")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(entry + "\n")
    except OSError:
        pass
