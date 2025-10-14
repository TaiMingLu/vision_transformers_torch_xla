# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_convnext(var_name):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3 
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_convnext(var_name)


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    # Check if we're in TPU mode to avoid XLA compilation issues
    tpu_mode = False
    try:
        import os
        if os.environ.get('PJRT_DEVICE') == 'TPU':
            tpu_mode = True
    except:
        pass

    # Skip XLA sync before parameter iteration - causes long freezes

    if tpu_mode:
        print("TPU mode: Using name-only parameter grouping to avoid XLA compilation")
        
        # Step 1: Get all parameter names without touching tensors
        param_names = []
        print("TPU mode: Collecting parameter names...")
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_names.append(name)
        
        print(f"TPU mode: Found {len(param_names)} trainable parameters")
        
        # Step 2: Group parameters by name patterns only
        print("TPU mode: Grouping parameters by names...")
        for i, name in enumerate(param_names):
            if i % 50 == 0:
                print(f"Processing parameter {i+1}/{len(param_names)}: {name}")
                
            # Use parameter name patterns to infer properties
            is_bias_or_1d = (name.endswith(".bias") or 
                           name.endswith(".weight") and ("norm" in name.lower() or "bn" in name.lower()) or
                           name in skip_list)
            
            if is_bias_or_1d:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
                
            if get_num_layer is not None:
                layer_id = get_num_layer(name)
                group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_group_names:
                if get_layer_scale is not None:
                    scale = get_layer_scale(layer_id)
                else:
                    scale = 1.

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            
            parameter_group_names[group_name]["params"].append(name)
        
        print("TPU mode: Parameter names grouped successfully")
        
        # Step 3: Now map names to actual tensors in one batch
        print("TPU mode: Mapping names to tensors...")
        param_dict = dict(model.named_parameters())
        for group_name, group_info in parameter_group_names.items():
            for param_name in group_info["params"]:
                if param_name in param_dict:
                    parameter_group_vars[group_name]["params"].append(param_dict[param_name])
        
        param_count = len(param_names)
        print(f"TPU mode: All {param_count} parameters mapped successfully")
        
    else:
        # Original logic for non-TPU
        print("Non-TPU mode: Standard parameter iteration...")
        param_count = 0
        for name, param in model.named_parameters():
            param_count += 1
            
            if param_count % 50 == 0:
                print(f"Processing parameter {param_count}: {name}")
            
            if not param.requires_grad:
                continue
            
            is_bias_or_1d = (len(param.shape) == 1 or name.endswith(".bias") or name in skip_list)
            
            if is_bias_or_1d:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay" 
                this_weight_decay = weight_decay
                
            if get_num_layer is not None:
                layer_id = get_num_layer(name)
                group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_group_names:
                if get_layer_scale is not None:
                    scale = get_layer_scale(layer_id)
                else:
                    scale = 1.

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
            
    print(f"Parameter iteration completed! Processed {param_count} parameters.")
    
    # Skip final XLA sync - let the training loop handle synchronization
    
    # For TPU, minimize printing to avoid sync issues
    if tpu_mode:
        print(f"TPU mode: Created {len(parameter_group_names)} parameter groups, returning optimizer groups...")
    else:
        print("About to print param groups...")
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        print("About to return parameter groups...")
    
    result = list(parameter_group_vars.values())
    print("Parameter groups created successfully")
    return result


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # if weight_decay and filter_bias_and_bn:
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
