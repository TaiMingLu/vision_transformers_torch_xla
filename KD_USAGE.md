# Knowledge Distillation Usage Guide

This document explains how to use the knowledge distillation (KD) functionality that has been added to the training script.

## Overview

Knowledge distillation allows a smaller "student" model to learn from a larger, pre-trained "teacher" model. The student model is trained using a combination of:
1. Standard cross-entropy loss with ground truth labels
2. Distillation loss that matches the student's predictions to the teacher's soft predictions

## Command Line Arguments

Five new arguments have been added for knowledge distillation:

- `--kd`: Enable knowledge distillation (default: False)
- `--teacher_path`: Path to the teacher model checkpoint (required when --kd is True)
- `--teacher_arch`: Teacher model architecture (required when --kd is True)
- `--kd_alpha`: Weight for combining CE loss and KD loss (default: 0.7)
- `--kd_temperature`: Temperature for distillation loss (default: 4.0)

## Usage Examples

### Basic Knowledge Distillation

Train a ViT-Tiny student model using a ViT-Small teacher:

```bash
python main.py \
    --model vit_tiny_patch16_224 \
    --kd true \
    --teacher_path /path/to/teacher/checkpoint.pth \
    --teacher_arch vit_small_patch16_224 \
    --kd_alpha 0.7 \
    --kd_temperature 4.0 \
    --epochs 100 \
    --batch_size 64
```

### ConvNeXt Distillation

Train a ConvNeXt-Small student using a ConvNeXt-Base teacher:

```bash
python main.py \
    --model convnext_small \
    --kd true \
    --teacher_path /path/to/convnext_base_teacher.pth \
    --teacher_arch convnext_base \
    --kd_alpha 0.6 \
    --kd_temperature 3.0 \
    --epochs 200 \
    --batch_size 32
```

### Cross-Architecture Distillation

Train a ViT student using a ConvNeXt teacher:

```bash
python main.py \
    --model vit_small_patch16_224 \
    --kd true \
    --teacher_path /path/to/convnext_large_teacher.pth \
    --teacher_arch convnext_large \
    --kd_alpha 0.8 \
    --kd_temperature 5.0 \
    --epochs 150 \
    --batch_size 48
```

## Parameter Explanations

### `--kd_alpha`
Controls the balance between standard training and distillation:
- `0.0`: Only standard cross-entropy loss (no distillation)
- `1.0`: Only distillation loss (no ground truth)
- `0.7` (default): 30% ground truth loss + 70% distillation loss

Higher values put more emphasis on learning from the teacher.

### `--kd_temperature`
Controls how "soft" the probability distributions are:
- Higher temperature (4-6): Softer distributions, more emphasis on learning teacher's uncertainty
- Lower temperature (2-3): Sharper distributions, more focused learning

## Implementation Details

### Model Wrapping
- The student model is wrapped in a `StudentWithDistillation` class
- During training: Returns both student and teacher logits
- During evaluation: Returns only student logits

### Loss Computation
- **Training**: Combined loss = (1-α) × CE_loss + α × KD_loss
- **Evaluation**: Standard cross-entropy loss only

### Teacher Model
- Loaded once at the beginning of training
- Set to evaluation mode and frozen (no gradient updates)
- Teacher inference is done with `torch.no_grad()` for efficiency

## Distributed Training Support

Knowledge distillation works seamlessly with distributed training:
- Teacher model is replicated across all GPUs
- Only student model parameters are optimized
- Evaluation correctly uses only the student model

## Model EMA Support

When using model EMA (`--model_ema true`):
- EMA is applied only to the student model
- Teacher model remains unchanged throughout training

## Testing

Run the test script to verify the setup:

```bash
python test_kd.py
```

This will test the KD implementation with dummy data and verify all components work correctly.

## Tips for Best Results

1. **Teacher Quality**: Use a well-trained, high-accuracy teacher model
2. **Architecture Match**: Similar architectures often work better, but cross-arch distillation can work
3. **Temperature Tuning**: Start with 4.0 and adjust based on results
4. **Alpha Tuning**: Common values are 0.6-0.8, higher for more challenging datasets
5. **Learning Rate**: You may need to adjust the learning rate when using KD

## Troubleshooting

### Common Issues

1. **Teacher path not found**: Ensure the teacher checkpoint path is correct
2. **Architecture mismatch**: Verify the teacher architecture matches the checkpoint
3. **Memory issues**: Teacher model doubles GPU memory usage
4. **Slow training**: Teacher inference adds computational overhead

### Memory Optimization

If running into memory issues:
- Reduce batch size
- Use gradient accumulation (`--update_freq`)
- Consider using CPU for teacher model (requires code modification) 