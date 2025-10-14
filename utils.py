# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path

import torch
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter


### Added for TPU
def init_distributed_mode_xla(args):
    """
    Initialize PyTorch/XLA distributed across multi-host TPUs.
    Uses the 'xla' backend and PJRT rendezvous (init_method='xla://').
    Scrubs NCCL/TCP env so we never fall back to TCPStore:12355.
    """
    # 1) Nuke GPU/NCCL env that can force TCP rendezvous
    for k in [
        "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",
        "LOCAL_RANK", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES",
        "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_DEBUG",
    ]:
        os.environ.pop(k, None)

    # 2) Bring in XLA backend and core modules
    import torch_xla.distributed.xla_backend  # registers the 'xla' backend
    import torch_xla.core.xla_model as xm
    
    # 3) For multi-host TPU, we need to handle initialization more carefully
    try:
        # First, let's check if we're already in a distributed environment
        if dist.is_initialized():
            print("[XLA] Distributed already initialized, using existing setup")
        else:
            # Initialize with timeout for multi-host coordination
            print("[XLA] Initializing distributed training...")
            
            # Try with a longer timeout for v4-64 multi-host setup
            import datetime
            timeout = datetime.timedelta(seconds=1800)  # 30 minutes for large TPU setup
            
            dist.init_process_group(
                backend="xla", 
                init_method="xla://",
                timeout=timeout
            )
        
        # 4) Fill args like the GPU path does, but from XLA/PG
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.gpu = 0
        args.dist_backend = "xla"
        args.distributed = True

        # 5) Standard print control on rank0
        setup_for_distributed(args.rank == 0)

        # Nice visibility
        dev = xm.xla_device()
        import socket
        print(f"[XLA] init ok | host={socket.gethostname()} rank={args.rank}/{args.world_size} device={dev}")
        
        # Additional TPU-specific info
        print(f"[XLA] TPU ordinal: {xm.get_ordinal()}, Local ordinal: {xm.get_local_ordinal()}")
        print(f"[XLA] World size: {xm.xrt_world_size()}")
        
    except Exception as e:
        print(f"[XLA] Failed to initialize distributed training: {e}")
        print("[XLA] Falling back to single-device training")
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.dist_backend = "xla"
        args.distributed = False
        setup_for_distributed(True)
        
        # Still set up XLA device
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        import socket
        print(f"[XLA] Single device mode | host={socket.gethostname()} device={dev}")

### End of TPU




class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    ### Added for TPU
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        import os
        dev = None
        if os.environ.get('PJRT_DEVICE') == 'TPU':
            dev = torch.device('xla')
        elif torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=dev)
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    ### End of TPU




    # def synchronize_between_processes(self):
    #     """
    #     Warning: does not synchronize the deque!
    #     """
    #     if not is_dist_avail_and_initialized():
    #         return
    #     t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
    #     dist.barrier()
    #     dist.all_reduce(t)
    #     t = t.tolist()
    #     self.count = int(t[0])
    #     self.total = t[1]



    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            try:
                init_kwargs = {
                    'project': args.project,
                    'entity': 'ttl',  # Add entity explicitly
                    'config': vars(args)  # Convert args to dict for better serialization
                }
                # Use experiment name as run name if provided, otherwise create a meaningful name
                if hasattr(args, 'experiment') and args.experiment is not None:
                    init_kwargs['name'] = args.experiment
                else:
                    # Create a meaningful run name based on model and key parameters
                    run_name_parts = [args.model]
                    if hasattr(args, 'batch_size'):
                        run_name_parts.append(f"bs{args.batch_size}")
                    if hasattr(args, 'lr'):
                        run_name_parts.append(f"lr{args.lr}")
                    if hasattr(args, 'epochs'):
                        run_name_parts.append(f"ep{args.epochs}")
                    if hasattr(args, 'kd') and args.kd:
                        run_name_parts.append("kd")
                    
                    init_kwargs['name'] = "_".join(run_name_parts)
                
                # Handle offline mode
                if hasattr(args, 'wandb_mode') and args.wandb_mode == 'offline':
                    init_kwargs['mode'] = 'offline'
                    if hasattr(args, 'wandb_dir') and args.wandb_dir:
                        init_kwargs['dir'] = args.wandb_dir
                    print(f"Initializing W&B in offline mode. Files will be saved to: {init_kwargs.get('dir', '~/.wandb')}")
                
                print(f"W&B init kwargs: {init_kwargs}")
                self._wandb.init(**init_kwargs)
                
                # Print run info
                if hasattr(args, 'wandb_mode') and args.wandb_mode == 'offline':
                    print(f"W&B offline run started. Run ID: {self._wandb.run.id}")
                    print(f"Run name: {self._wandb.run.name}")
                    print(f"Project: {self._wandb.run.project}")
                    print(f"Entity: {self._wandb.run.entity}")
                    print(f"To sync later, use: wandb sync {self._wandb.run.dir}")
                else:
                    print(f"W&B online run started. URL: {self._wandb.run.url}")
            except Exception as e:
                print(f"Failed to initialize W&B: {e}")
                raise

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        if 'n_parameters' in metrics:
            self._wandb.summary['n_parameters'] = metrics['n_parameters']
            metrics.pop('n_parameters')

        # Log current epoch
        if 'epoch' in metrics:
            self._wandb.log({'epoch': metrics['epoch']}, commit=False)
            metrics.pop('epoch')

        # Log all metrics with proper grouping
        for k, v in metrics.items():
            if v is not None:  # Skip None values
                if 'train' in k:
                    self._wandb.log({f'Global Train/{k}': v}, commit=False)
                elif 'test' in k:
                    self._wandb.log({f'Global Test/{k}': v}, commit=False)
                else:
                    self._wandb.log({k: v}, commit=False)

        # Commit all logged metrics
        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')
    
    def finish(self):
        """Finish the W&B run properly"""
        if self._wandb.run is not None:
            self._wandb.finish()
            print("W&B run finished successfully")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        # For TPU: move tensors to CPU before saving
        if len(args) >= 1 and isinstance(args[0], dict):
            data_to_save = args[0]
            # Check if we have XLA tensors and move them to CPU
            try:
                import torch_xla.core.xla_model as xm
                print("Moving checkpoint tensors to CPU for saving...", flush=True)
                
                def move_to_cpu(obj):
                    if isinstance(obj, dict):
                        return {k: move_to_cpu(v) for k, v in obj.items()}
                    elif hasattr(obj, 'cpu'):  # torch.Tensor
                        return obj.cpu()
                    else:
                        return obj
                
                cpu_data = move_to_cpu(data_to_save)
                print("Tensors moved to CPU, starting torch.save...", flush=True)
                torch.save(cpu_data, *args[1:], **kwargs)
                print("torch.save completed successfully", flush=True)
                
            except ImportError:
                # Not using XLA, save normally
                torch.save(*args, **kwargs)
        else:
            torch.save(*args, **kwargs)


def init_distributed_mode(args):

    # Check if we're using TPU
    if hasattr(args, 'tpu') and args.tpu:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_backend
        
        # For TPU, we use XLA's distributed setup
        args.device = xm.xla_device()
        args.world_size = xm.xrt_world_size()
        args.rank = xm.get_ordinal()
        args.gpu = args.rank  # For compatibility
        args.distributed = args.world_size > 1
        
        if args.distributed:
            # Initialize XLA distributed backend
            torch.distributed.init_process_group(
                backend='xla',
                init_method='xla://'
            )
            print(f'TPU distributed init (rank {args.rank}/{args.world_size})', flush=True)
        
        setup_for_distributed(args.rank == 0)
        return

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        
        print(f"SLURM setup: rank={args.rank}, world_size={args.world_size}, gpu={args.gpu}")
        print(f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}, SLURM_NNODES={os.environ.get('SLURM_NNODES')}, SLURM_NTASKS_PER_NODE={os.environ.get('SLURM_NTASKS_PER_NODE')}")
        print(f"dist_url={args.dist_url}")
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, device='cuda'):
        # Make scaler device-aware to support both CUDA and TPU
        if device == 'cuda' or torch.cuda.is_available():
            self._scaler = torch.amp.GradScaler('cuda')
        else:
            # For TPU or other devices, create a dummy scaler that does nothing
            self._scaler = None
        self._device = device

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        if self._scaler is not None:
            # CUDA AMP path
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = get_grad_norm_(parameters)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                norm = None
        else:
            # TPU or non-AMP path - no scaling needed
            loss.backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters) if parameters else None
                # Note: optimizer.step() is handled by the engine for TPU
            else:
                norm = None
        return norm

    def state_dict(self):
        if self._scaler is not None:
            return self._scaler.state_dict()
        else:
            return {}  # Return empty dict for TPU/non-AMP mode

    def load_state_dict(self, state_dict):
        if self._scaler is not None and state_dict:
            self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    # XLA synchronization for TPU before saving
    is_tpu = getattr(args, 'tpu', False)
    if is_tpu:
        try:
            import torch_xla.core.xla_model as xm
            print("Synchronizing XLA before checkpoint save...", flush=True)
            xm.mark_step()  # Ensure all operations are complete
            xm.wait_device_ops()  # Wait for all device operations to finish
            print("XLA synchronization completed", flush=True)
        except ImportError:
            pass
    
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        print("Building checkpoint dictionary...", flush=True)
        
        print("Getting model state_dict...", flush=True)
        model_state = model_without_ddp.state_dict()
        print("Getting optimizer state_dict...", flush=True)
        optimizer_state = optimizer.state_dict()
        
        to_save = {
            'model': model_state,
            'optimizer': optimizer_state,
            'epoch': epoch,
            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,  # Handle None properly
            'args': args,
        }
        print("Basic checkpoint dict created", flush=True)

        if model_ema is not None:
            print("Getting model_ema state...", flush=True)
            to_save['model_ema'] = get_state_dict(model_ema)
            print("Model_ema added to checkpoint", flush=True)

        print(f"About to call save_on_master for {checkpoint_path}...", flush=True)
        save_on_master(to_save, checkpoint_path)
        print(f"save_on_master completed for {checkpoint_path}", flush=True)
    
    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str): # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'])
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'])
            if 'scaler' in checkpoint and loss_scaler is not None:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
