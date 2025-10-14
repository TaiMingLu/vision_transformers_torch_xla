#!/usr/bin/env python3

"""
Test script to verify knowledge distillation implementation.
This script tests the basic functionality without running full training.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_model

def test_kd_setup():
    """Test the knowledge distillation setup"""
    print("Testing Knowledge Distillation Setup...")
    
    # Create student and teacher models
    student = create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=1000,
        global_pool='avg',
        drop_path_rate=0.1,
    )
    
    teacher = create_model(
        'vit_small_patch16_224',
        pretrained=False,
        num_classes=1000,
        global_pool='avg',
        drop_path_rate=0.0,
    )
    
    print(f"Student model created: {sum(p.numel() for p in student.parameters())} parameters")
    print(f"Teacher model created: {sum(p.numel() for p in teacher.parameters())} parameters")
    
    # Test StudentWithDistillation wrapper
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
    
    # Test DistillationLoss
    class DistillationLoss(nn.Module):
        def __init__(self, base_criterion, alpha=0.7, temperature=4.0):
            super().__init__()
            self.base_criterion = base_criterion
            self.alpha = alpha
            self.temperature = temperature
            self.kl_div = nn.KLDivLoss(reduction='batchmean')
            
        def forward(self, outputs, targets):
            if isinstance(outputs, tuple):
                # Knowledge distillation mode
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
                
                print(f"CE Loss: {ce_loss.item():.4f}, KD Loss: {kd_loss.item():.4f}, Total: {total_loss.item():.4f}")
                return total_loss
            else:
                # Standard mode
                return self.base_criterion(outputs, targets)
    
    # Create wrapped model and loss
    wrapped_model = StudentWithDistillation(student, teacher)
    base_criterion = nn.CrossEntropyLoss()
    kd_criterion = DistillationLoss(base_criterion, alpha=0.7, temperature=4.0)
    
    # Test with dummy data
    batch_size = 4
    input_size = 224
    num_classes = 1000
    
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    dummy_targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\nTesting with dummy data: {dummy_input.shape}")
    
    # Test training mode (should return tuple)
    wrapped_model.train()
    outputs = wrapped_model(dummy_input)
    print(f"Training mode output type: {type(outputs)}")
    if isinstance(outputs, tuple):
        print(f"Student logits shape: {outputs[0].shape}, Teacher logits shape: {outputs[1].shape}")
    
    # Test loss computation
    loss = kd_criterion(outputs, dummy_targets)
    print(f"KD Loss computed successfully: {loss.item():.4f}")
    
    # Test evaluation mode (should return single tensor)
    wrapped_model.eval()
    outputs_eval = wrapped_model(dummy_input)
    print(f"Evaluation mode output type: {type(outputs_eval)}")
    print(f"Evaluation output shape: {outputs_eval.shape}")
    
    # Test standard loss in eval mode
    loss_eval = kd_criterion(outputs_eval, dummy_targets)
    print(f"Standard loss in eval mode: {loss_eval.item():.4f}")
    
    print("\n✅ Knowledge Distillation setup test passed!")

if __name__ == "__main__":
    test_kd_setup() 