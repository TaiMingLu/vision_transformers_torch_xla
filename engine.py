# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import time

### Added for TPU
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, tpu: bool = False, log_freq: int = 10):
    import time
    import datetime
    print(f"🎯 train_one_epoch called for epoch {epoch}", flush=True)
    function_start_time = time.time()
    
    model.train(True)

    if log_writer is not None:
        log_writer.set_step(epoch * num_training_steps_per_epoch * update_freq)

    if tpu:
        import torch_xla
        import torch_xla.core.xla_model as xm

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 10
    
    # DataLoader is already wrapped with MpDeviceLoader in main.py for TPU
    device_loader = data_loader

    print("About to zero_grad optimizer...", flush=True)
    optimizer.zero_grad()
    step_timer = time.time()
    print("About to start data loader loop...", flush=True)
    rank = utils.get_rank()

    def _train_log(message: str) -> None:
        stamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        print(f"[train][{stamp}] rank={rank} {message}", flush=True)

    if tpu:
        total_batches = len(data_loader) if hasattr(data_loader, "__len__") else "unknown"
        _train_log(f"entering epoch {epoch} total_batches={total_batches}")

    # For TPU, bypass metric_logger.log_every to reduce overhead
    if tpu:
        print("TPU mode: Using direct DataLoader enumeration to avoid metric_logger overhead", flush=True)
        dataloader_iterator = enumerate(device_loader)
    else:
        dataloader_iterator = enumerate(metric_logger.log_every(device_loader, print_freq, header))
    
    for data_iter_step, (samples, targets) in dataloader_iterator:
        iteration_start = time.time()
        
        dataloader_time = time.time() - (getattr(train_one_epoch, '_last_iteration_end', iteration_start))
        # print(f"🚀 Iteration: {data_iter_step}", flush=True)
        print(f"🚀 Starting data_iter_step {data_iter_step} (DataLoader iteration took {dataloader_time:.3f}s)...", flush=True)
        
        # Debug resource usage if DataLoader is slow
        # if dataloader_time > 30.0:  # If > 30s, investigate
        #     try:
        #         import psutil
        #         import os
        #         process = psutil.Process(os.getpid())
        #         print(f"🚨 SLOW DATALOADER DEBUG: RAM={process.memory_info().rss / (1024**3):.2f}GB, "
        #               f"Open files={len(process.open_files())}, Threads={process.num_threads()}", flush=True)
        #     except:
        #         print(f"🚨 SLOW DATALOADER: {dataloader_time:.1f}s delay detected", flush=True)
        step = data_iter_step // update_freq
        # print(f"Calculated step = {step}, num_training_steps_per_epoch = {num_training_steps_per_epoch}", flush=True)
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # print(f"Global iteration it = {it}", flush=True)
        
        # Check for potential issues every 50 iterations
        # if data_iter_step % 50 == 0 and tpu:
        #     print(f"TPU mode: Processing iteration {data_iter_step}, step {step}, samples shape: {samples.shape if hasattr(samples, 'shape') else 'unknown'}", flush=True)

        # print(f"About to update LR/WD schedules...", flush=True)
        schedule_start = time.time()
        # Update per-iteration LR & WD before backward
        if (lr_schedule_values is not None or wd_schedule_values is not None) and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group.get("weight_decay", None) is not None:
                    param_group["weight_decay"] = wd_schedule_values[it]
        schedule_time = time.time() - schedule_start
        # print(f"✅ LR/WD schedule update completed in {schedule_time:.3f}s", flush=True)
        # print(f"About to enter forward/backward section, use_amp={use_amp}, tpu={tpu}...", flush=True)
        
        # Forward / backward / step
        if use_amp and not tpu:
            print(f"Taking CUDA AMP path...", flush=True)
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        elif tpu:
            # print(f"Taking TPU path...", flush=True)
            import time
            step_start = time.time()
            # print(f"TPU mode: Starting XLA step for iteration {data_iter_step}...", flush=True)
            # Verify we're actually on TPU device
            import torch_xla.core.xla_model as xm
            current_device = xm.xla_device()
            # if data_iter_step == 0:  # Only print device info for first iteration
            #     print(f"TPU mode: Current device = {current_device}, device type = {current_device.type}", flush=True)
            
            # print(f"About to apply mixup if present...", flush=True)
            # For TPU: put everything in XLA context
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            
            # print(f"About to enter torch_xla.step() context...", flush=True)
            xla_step_start = time.time()
            with torch_xla.step():
                # MpDeviceLoader already moves data to device, so we skip manual transfer
                if hasattr(samples, 'device') and 'xla' in str(samples.device):
                    # print(f"✅ Data already on XLA device: {samples.device}", flush=True)
                    pass
                else:
                    # print(f"⚠️ Data not on XLA device, moving manually...", flush=True)
                    data_move_start = time.time()
                    samples = samples.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    data_move_time = time.time() - data_move_start
                    # print(f"✅ Data moved to device in {data_move_time:.3f}s", flush=True)
                
                # Verify data is on TPU
                # if data_iter_step == 0:  # Only print for first iteration to avoid spam
                #     print(f"TPU mode: samples.device = {samples.device}", flush=True)
                #     print(f"TPU mode: targets.device = {targets.device}", flush=True)
                #     print(f"TPU mode: model device = {next(model.parameters()).device}", flush=True)
                
                # print(f"About to call model forward...", flush=True)
                forward_start = time.time()
                output = model(samples)
                forward_time = time.time() - forward_start
                # print(f"✅ Model forward completed in {forward_time:.3f}s, about to compute loss...", flush=True)
                
                loss_start = time.time()
                loss = criterion(output, targets)
                loss = loss / update_freq
                loss_time = time.time() - loss_start
                # print(f"✅ Loss computed in {loss_time:.3f}s: {loss.item()}, about to backward...", flush=True)
                
                backward_start = time.time()
                loss.backward()
                backward_time = time.time() - backward_start
                # print(f"✅ Backward completed in {backward_time:.3f}s", flush=True)
                
                if (data_iter_step + 1) % update_freq == 0:
                    # Apply gradient clipping if specified
                    if max_norm is not None and max_norm > 0:
                        clip_start = time.time()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        clip_time = time.time() - clip_start
                        # print(f"✅ Gradient clipping in {clip_time:.3f}s", flush=True)
                    
                    # print(f"About to run optimizer step...", flush=True)
                    optim_start = time.time()
                    if tpu:
                        _train_log(f"iter={data_iter_step} step={step} optimizer_step start")
                    xm.optimizer_step(optimizer, barrier=True)  # does all-reduce + mark_step
                    if tpu:
                        _train_log(
                            f"iter={data_iter_step} step={step} optimizer_step done in {time.time() - optim_start:.2f}s"
                        )
                    optimizer.zero_grad()
                    optim_time = time.time() - optim_start
                    # print(f"✅ Optimizer step completed in {optim_time:.3f}s", flush=True)
            
            xla_step_time = time.time() - xla_step_start
            # print(f"🔥 Total XLA step time: {xla_step_time:.3f}s", flush=True)

            # Move EMA outside XLA context to avoid graph complications
            if (data_iter_step + 1) % update_freq == 0 and model_ema is not None:
                ema_start = time.time()
                model_ema.update(model)
                ema_time = time.time() - ema_start
                # print(f"✅ EMA update in {ema_time:.3f}s", flush=True)

            # Print timing breakdown for first few iterations and update end time for all iterations
            # if data_iter_step < 20:
            #     step_end = time.time()
            #     print(f"TPU mode: Iteration {data_iter_step} completed in {step_end - step_start:.2f}s")
            #     print(f"📝 End of iteration {data_iter_step} - about to continue loop...", flush=True)
            
            # Always update the last iteration end time for accurate DataLoader timing
            train_one_epoch._last_iteration_end = time.time()
            total_iteration_time = train_one_epoch._last_iteration_end - iteration_start

            if tpu:
                should_log_step = data_iter_step < 5 or (data_iter_step + 1) % 100 == 0
                if should_log_step:
                    _train_log(
                        f"iter={data_iter_step} step={step} loss={float(loss.item())} step_time={total_iteration_time:.2f}s"
                    )
            
            # Manual logging for TPU (since we bypass metric_logger.log_every)
            if tpu and data_iter_step % print_freq == 0:
                # Calculate metrics for logging
                loss_value = loss.item() if hasattr(loss, 'item') else loss
                max_lr = 0.0
                min_lr = 10.0
                for group in optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])
                
                eta_seconds = (num_training_steps_per_epoch - data_iter_step) * (total_iteration_time if data_iter_step > 0 else 1.0)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(f"Epoch: [{epoch}] [{data_iter_step}/{num_training_steps_per_epoch}] "
                      f"eta: {eta_string} loss: {loss_value:.4f} lr: {max_lr:.6f} "
                      f"time: {total_iteration_time:.4f} data: {dataloader_time:.4f}", flush=True)
            
            # Monitor memory every 50 iterations to catch issues
            # if data_iter_step % 50 == 0:
            #     try:
            #         mem_info = xm.get_memory_info(current_device)
            #         print(f"TPU mode: Iteration {data_iter_step}, TPU memory: {mem_info}")
            #     except Exception as e:
            #         print(f"TPU mode: Iteration {data_iter_step}, memory check failed: {e}")
                
        elif use_amp and not tpu:
            # CUDA AMP path (unchanged)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss = loss / update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        else:
            # Non-TPU path: handle data and forward pass first
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            output = model(samples)
            loss = criterion(output, targets)
            
            # Full precision CPU/GPU path
            loss = loss / update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        # Flush pending work per backend
        # Skip extra sync - xm.optimizer_step already handles TPU sync
        if not tpu and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Metrics
        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        # Only do expensive metric logging based on log_freq to reduce overhead
        # if data_iter_step == 0:
        #     print(f"🔧 Using log_freq={log_freq} for logging", flush=True)
        if data_iter_step % log_freq == 0 or data_iter_step < 5:
            # print(f"🔧 Starting metric updates...", flush=True)
            metric_start = time.time()
            
            metric_logger.update(loss=loss.item())
            if class_acc is not None:
                metric_logger.meters['class_acc'].update(class_acc.item(), n=samples.size(0))

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            metric_time = time.time() - metric_start
            # print(f"✅ Metric logger updated in {metric_time:.3f}s", flush=True)
        else:
            # Fast logging without synchronization
            lr = optimizer.param_groups[0]["lr"]

        if log_writer is not None:
            log_start = time.time()
            log_writer.update(loss=loss.item(), lr=lr)
            log_time = time.time() - log_start
            # print(f"✅ Log writer updated in {log_time:.3f}s", flush=True)
            
        # Wandb logging for TPU training - same frequency as console logging
        if wandb_logger is not None and (data_iter_step % log_freq == 0 or data_iter_step < 5):
            wandb_start = time.time()
            wandb_metrics = {
                'train/loss': loss.item(),
                'train/learning_rate': lr,
                'train/epoch': epoch,
                'train/step': start_steps + data_iter_step if start_steps else data_iter_step
            }
            if class_acc is not None:
                wandb_metrics['train/accuracy'] = class_acc.item()
            wandb_logger._wandb.log(wandb_metrics)
            wandb_time = time.time() - wandb_start
            # print(f"✅ Wandb logged in {wandb_time:.3f}s", flush=True)

    # end epoch
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    function_end_time = time.time()
    function_total_time = function_end_time - function_start_time
    print(f"🏁 train_one_epoch total time: {function_total_time:.2f}s", flush=True)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
### End of TPU



### Added for TPU
@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, tpu: bool = False):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    original_device = None
    restored_to_cpu = False
    try:
        param = next(model.parameters())
        original_device = param.device
    except StopIteration:
        original_device = torch.device('cpu')

    if tpu and original_device.type != device.type:
        print(f"[eval] moving model from {original_device} to {device}", flush=True)
        model = model.to(device)
        restored_to_cpu = True

    model.eval()

    if tpu:
        import torch_xla.core.xla_model as xm

    print("Starting validation loop...", flush=True)
    print(f"Validation DataLoader type: {type(data_loader)}", flush=True)
    print(f"Validation DataLoader batch count: {len(data_loader) if hasattr(data_loader, '__len__') else 'unknown'}", flush=True)
    
    # Add timing around the first enumerate call for validation
    enumerate_start = time.time()
    print("About to call enumerate(metric_logger.log_every(...)) for validation...", flush=True)
    
    # For TPU, bypass metric_logger.log_every to reduce overhead  
    if tpu:
        print("TPU mode: Using direct validation DataLoader enumeration to avoid metric_logger overhead", flush=True)
        validation_iterator = enumerate(data_loader)
    else:
        validation_iterator = enumerate(metric_logger.log_every(data_loader, 10, header))
    
    with torch.no_grad():
        for batch_idx, batch in validation_iterator:
            if batch_idx == 0:
                enumerate_time = time.time() - enumerate_start
                print(f"📊 First validation enumerate() call took {enumerate_time:.2f}s", flush=True)
            
            validation_iteration_start = time.time()
            val_dataloader_time = time.time() - (getattr(evaluate, '_last_val_iteration_end', validation_iteration_start))
            if batch_idx % 50 == 0:  # Print every 50 batches to avoid spam
                print(f"🔍 Starting validation batch {batch_idx} (DataLoader iteration took {val_dataloader_time:.3f}s)...", flush=True)
            images = batch[0]
            target = batch[-1]

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if use_amp and torch.cuda.is_available() and not tpu:
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, target)
            else:
                output = model(images)
                loss = criterion(output, target)

            if tpu:
                xm.mark_step()

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            
            # Track timing for next iteration
            evaluate._last_val_iteration_end = time.time()
            val_iteration_total_time = evaluate._last_val_iteration_end - validation_iteration_start
            
            # Manual logging for TPU (since we bypass metric_logger.log_every for validation)
            if tpu and batch_idx % 10 == 0:  # Log every 10 validation batches
                print(f"Test: [{batch_idx}/3] loss: {loss.item():.4f} "
                      f"acc1: {acc1.item():.4f} acc5: {acc5.item():.4f} "
                      f"time: {val_iteration_total_time:.4f} data: {val_dataloader_time:.4f}", flush=True)
            elif batch_idx % 50 == 0:  # Print every 50 batches to avoid spam for non-TPU
                print(f"✅ Validation batch {batch_idx} completed in {val_iteration_total_time:.3f}s", flush=True)

    metric_logger.synchronize_between_processes()
    if tpu and restored_to_cpu:
        print(f"[eval] moving model back to {original_device}", flush=True)
        model.to(original_device)
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['acc1'], top5=metric_logger.meters['acc5'],
                  losses=metric_logger.meters['loss']))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
### End of TPU








# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
#                     wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
#                     num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
#     model.train(True)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10

#     optimizer.zero_grad()

#     step_timer = time.time()
#     for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         step = data_iter_step // update_freq
#         if step >= num_training_steps_per_epoch:
#             continue
#         it = start_steps + step  # global training iteration
#         # Update LR & WD for the first acc
#         if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
#             for i, param_group in enumerate(optimizer.param_groups):
#                 if lr_schedule_values is not None:
#                     param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
#                 if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                     param_group["weight_decay"] = wd_schedule_values[it]

#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)

#         if use_amp:
#             with torch.cuda.amp.autocast():
#                 output = model(samples)
#                 loss = criterion(output, targets)
#         else: # full precision
#             output = model(samples)
#             loss = criterion(output, targets)

#         loss_value = loss.item()

#         if not math.isfinite(loss_value): # this could trigger if using AMP
#             print("Loss is {}, stopping training".format(loss_value))
#             assert math.isfinite(loss_value)

#         if use_amp:
#             # this attribute is added by timm on one optimizer (adahessian)
#             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#             loss /= update_freq
#             grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
#                                     parameters=model.parameters(), create_graph=is_second_order,
#                                     update_grad=(data_iter_step + 1) % update_freq == 0)
#             if (data_iter_step + 1) % update_freq == 0:
#                 optimizer.zero_grad()
#                 if model_ema is not None:
#                     model_ema.update(model)
#         else: # full precision
#             loss /= update_freq
#             loss.backward()
#             if (data_iter_step + 1) % update_freq == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 if model_ema is not None:
#                     model_ema.update(model)

#         torch.cuda.synchronize()

#         if mixup_fn is None:
#             class_acc = (output.max(-1)[-1] == targets).float().mean()
#         else:
#             class_acc = None
#         metric_logger.update(loss=loss_value)
#         metric_logger.update(class_acc=class_acc)
#         min_lr = 10.
#         max_lr = 0.
#         for group in optimizer.param_groups:
#             min_lr = min(min_lr, group["lr"])
#             max_lr = max(max_lr, group["lr"])

#         metric_logger.update(lr=max_lr)
#         metric_logger.update(min_lr=min_lr)
#         weight_decay_value = None
#         for group in optimizer.param_groups:
#             if group["weight_decay"] > 0:
#                 weight_decay_value = group["weight_decay"]
#         metric_logger.update(weight_decay=weight_decay_value)
#         if use_amp:
#             metric_logger.update(grad_norm=grad_norm)

#         if log_writer is not None:
#             log_writer.update(loss=loss_value, head="loss")
#             log_writer.update(class_acc=class_acc, head="loss")
#             log_writer.update(lr=max_lr, head="opt")
#             log_writer.update(min_lr=min_lr, head="opt")
#             log_writer.update(weight_decay=weight_decay_value, head="opt")
#             if use_amp:
#                 log_writer.update(grad_norm=grad_norm, head="opt")
#             log_writer.set_step()

#         if wandb_logger:
#             wandb_logger._wandb.log({
#                 'Rank-0 Batch Wise/train_loss': loss_value,
#                 'Rank-0 Batch Wise/train_max_lr': max_lr,
#                 'Rank-0 Batch Wise/train_min_lr': min_lr
#             }, commit=False)
#             if class_acc:
#                 wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
#             if use_amp:
#                 wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
#             wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

#         if (data_iter_step + 1) % 100 == 0:
#             elapsed = time.time() - step_timer
#             print(f"Time taken for steps {data_iter_step - 99} to {data_iter_step + 1}: {elapsed:.2f} seconds")
#             step_timer = time.time()

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# @torch.no_grad()
# def evaluate(data_loader, model, device, use_amp=False):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()
#     for batch in metric_logger.log_every(data_loader, 10, header):
#         images = batch[0]
#         target = batch[-1]

#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         if use_amp:
#             with torch.cuda.amp.autocast():
#                 output = model(images)
#                 loss = criterion(output, target)
#         else:
#             output = model(images)
#             loss = criterion(output, target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
