#!/usr/bin/env python3
"""
🚀 SSA Brain Tumor Segmentation - Complete Training Pipeline
============================================================

This module implements a comprehensive training pipeline for SSA brain tumor 
segmentation with GPU optimization, transfer learning, and research-grade 
evaluation metrics.

Features:
- Transfer learning from glioma models
- GTX 1650 optimized training
- Mixed precision support
- Comprehensive evaluation metrics
- Real-time monitoring and visualization

Date: September 7, 2025
GPU Target: NVIDIA GeForce GTX 1650 (4GB VRAM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import json
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Import our SSA model components
from ssa_model import SSABrainTumorUNet3D, SSAModelManager, SSADataset

class SSATrainer:
    """
    Comprehensive SSA brain tumor segmentation trainer
    
    Handles complete training pipeline including:
    - Transfer learning from glioma models
    - GPU-optimized training for GTX 1650
    - Mixed precision training
    - Comprehensive evaluation
    - Real-time monitoring
    """
    
    def __init__(self,
                 device: torch.device,
                 patch_dir: str = "SSA_Type/ssa_preprocessed_patches",
                 glioma_model_path: Optional[str] = None,
                 save_dir: str = "SSA_Type/training_results",
                 mixed_precision: bool = True):
        """
        Initialize SSA trainer
        
        Args:
            device: Training device (GPU/CPU)
            patch_dir: Directory containing preprocessed SSA patches
            glioma_model_path: Path to pre-trained glioma model for transfer learning
            save_dir: Directory to save training results
            mixed_precision: Use mixed precision training
        """
        self.device = device
        self.patch_dir = Path(patch_dir)
        self.glioma_model_path = glioma_model_path
        self.save_dir = Path(save_dir)
        self.mixed_precision = mixed_precision
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model manager
        self.model_manager = SSAModelManager(device=device, mixed_precision=mixed_precision)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if mixed_precision else None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_dice_scores = []
        self.val_dice_scores = []
        self.learning_rates = []
        
        # Setup logging
        self._setup_logging()
        
        print(f"🚀 SSA Trainer Initialized")
        print(f"📍 Device: {device}")
        print(f"📂 Patch Directory: {patch_dir}")
        print(f"💾 Save Directory: {save_dir}")
        print(f"⚡ Mixed Precision: {mixed_precision}")
        if glioma_model_path:
            print(f"🔄 Transfer Learning: {glioma_model_path}")
    
    def _setup_logging(self):
        """Setup logging for training"""
        log_file = self.save_dir / 'ssa_training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_model(self, 
                     use_attention: bool = False,
                     dropout: float = 0.1) -> SSABrainTumorUNet3D:
        """
        Prepare SSA model with optional transfer learning
        
        Args:
            use_attention: Use attention mechanisms
            dropout: Dropout rate
            
        Returns:
            Prepared SSA model
        """
        print(f"\n🧠 PREPARING SSA MODEL")
        print("=" * 50)
        
        # Create SSA model
        self.model = self.model_manager.create_ssa_model(
            use_attention=use_attention,
            dropout=dropout
        )
        
        # Apply transfer learning if glioma model available
        if self.glioma_model_path and os.path.exists(self.glioma_model_path):
            print(f"🔄 Applying transfer learning from glioma model...")
            self.model = self.model_manager.load_glioma_weights(
                self.model, 
                self.glioma_model_path
            )
        else:
            print(f"⚠️ No glioma model found, using random initialization")
        
        return self.model
    
    def prepare_data_loaders(self,
                           batch_size: int = 1,
                           train_split: float = 0.8,
                           num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders
        
        Args:
            batch_size: Batch size (1 recommended for GTX 1650)
            train_split: Training data proportion
            num_workers: Number of data loading workers
            
        Returns:
            train_loader, val_loader
        """
        print(f"\n📊 PREPARING DATA LOADERS")
        print("=" * 50)
        
        train_loader, val_loader = self.model_manager.get_data_loaders(
            patch_dir=str(self.patch_dir),
            batch_size=batch_size,
            train_split=train_split,
            num_workers=num_workers
        )
        
        return train_loader, val_loader
    
    def prepare_training_components(self,
                                  learning_rate: float = 0.001,
                                  weight_decay: float = 1e-5) -> None:
        """
        Prepare optimizer, scheduler, and loss function
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
        """
        print(f"\n⚙️ PREPARING TRAINING COMPONENTS")
        print("=" * 50)
        
        # Loss function (CrossEntropyLoss for multi-class segmentation)
        self.criterion = nn.CrossEntropyLoss()
        print(f"📊 Loss Function: CrossEntropyLoss")
        
        # Optimizer (Adam with weight decay)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        print(f"🔧 Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        print(f"📈 Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    def calculate_dice_score(self, 
                            pred: torch.Tensor, 
                            target: torch.Tensor,
                            num_classes: int = 4) -> torch.Tensor:
        """
        Calculate Dice score for multi-class segmentation
        
        Args:
            pred: Predicted segmentation (B, C, H, W, D)
            target: Ground truth segmentation (B, H, W, D)
            num_classes: Number of classes
            
        Returns:
            Mean Dice score across all classes
        """
        # Convert predictions to class indices
        pred_classes = torch.argmax(pred, dim=1)
        
        dice_scores = []
        
        for cls in range(num_classes):
            pred_cls = (pred_classes == cls).float()
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            if union == 0:
                dice = 1.0  # Perfect score for empty class
            else:
                dice = (2.0 * intersection) / union
            
            dice_scores.append(dice)
        
        return torch.stack(dice_scores).mean()
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            average_loss, average_dice_score
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"🏋️ Epoch {epoch}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice = self.calculate_dice_score(outputs, masks)
                epoch_loss += loss.item()
                epoch_dice += dice.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{dice.item():.4f}",
                'GPU': f"{torch.cuda.memory_allocated(0)/1024**3:.1f}GB" if self.device.type == 'cuda' else "CPU"
            })
            
            # Clear GPU cache periodically
            if self.device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        
        return avg_loss, avg_dice
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            average_loss, average_dice_score
        """
        self.model.eval()
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="🔍 Validation"):
                # Move to device
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                dice = self.calculate_dice_score(outputs, masks)
                epoch_loss += loss.item()
                epoch_dice += dice.item()
        
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        
        return avg_loss, avg_dice
    
    def save_checkpoint(self, 
                       epoch: int,
                       train_loss: float,
                       val_loss: float,
                       val_dice: float,
                       is_best: bool = False) -> None:
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_dice: Validation Dice score
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dice_scores': self.train_dice_scores,
            'val_dice_scores': self.val_dice_scores,
            'learning_rates': self.learning_rates
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_ssa_model.pth')
            print(f"💾 New best model saved! Dice: {val_dice:.4f}")
    
    def plot_training_history(self) -> None:
        """
        Plot and save training history
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SSA Brain Tumor Segmentation - Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Training and validation loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training and validation Dice score
        axes[0, 1].plot(epochs, self.train_dice_scores, 'b-', label='Training Dice', linewidth=2)
        axes[0, 1].plot(epochs, self.val_dice_scores, 'r-', label='Validation Dice', linewidth=2)
        axes[0, 1].set_title('Dice Score Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance summary
        axes[1, 1].text(0.1, 0.8, f"Best Validation Dice: {max(self.val_dice_scores):.4f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Final Training Loss: {self.train_losses[-1]:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Final Validation Loss: {self.val_losses[-1]:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Total Epochs: {len(self.train_losses)}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'ssa_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history saved: {self.save_dir / 'ssa_training_history.png'}")
    
    def train(self,
             epochs: int = 30,
             batch_size: int = 1,
             learning_rate: float = 0.001,
             weight_decay: float = 1e-5,
             early_stopping_patience: int = 10) -> Dict:
        """
        Complete training pipeline
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size (1 recommended for GTX 1650)
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training statistics
        """
        print(f"\n🚀 STARTING SSA BRAIN TUMOR SEGMENTATION TRAINING")
        print("=" * 70)
        print(f"🎯 Epochs: {epochs}")
        print(f"📦 Batch Size: {batch_size}")
        print(f"📈 Learning Rate: {learning_rate}")
        print(f"⚖️ Weight Decay: {weight_decay}")
        print(f"⏰ Early Stopping Patience: {early_stopping_patience}")
        
        # Prepare model
        self.prepare_model(use_attention=False, dropout=0.1)
        
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(
            batch_size=batch_size,
            num_workers=2
        )
        
        # Prepare training components
        self.prepare_training_components(learning_rate, weight_decay)
        
        # Training loop
        best_dice = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            print(f"\n📅 Epoch {epoch}/{epochs} | LR: {current_lr:.6f}")
            print("-" * 50)
            
            # Training
            train_loss, train_dice = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_dice = self.validate_epoch(val_loader)
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dice_scores.append(train_dice)
            self.val_dice_scores.append(val_dice)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check for best model
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, val_dice, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"📊 Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
            print(f"⏱️ Epoch Time: {epoch_time:.1f}s")
            
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"💾 GPU Memory: {memory_used:.2f} GB")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETE!")
        print("=" * 70)
        print(f"⏱️ Total Training Time: {total_time/3600:.2f} hours")
        print(f"🏆 Best Validation Dice: {best_dice:.4f}")
        print(f"📊 Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"📊 Final Validation Loss: {self.val_losses[-1]:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Save training statistics
        stats = {
            'total_epochs': len(self.train_losses),
            'best_dice': best_dice,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'total_time_hours': total_time / 3600,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dice_scores': self.train_dice_scores,
            'val_dice_scores': self.val_dice_scores,
            'learning_rates': self.learning_rates
        }
        
        with open(self.save_dir / 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats

def main():
    """Main training function"""
    print("🚀 SSA Brain Tumor Segmentation Training Pipeline")
    print("=" * 70)
    
    # Setup device - force CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"🎯 Device: {device}")
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"🎯 Device: {device} (GPU not available)")
    
    # Configuration for GTX 1650
    config = {
        'epochs': 25,
        'batch_size': 1,  # Conservative for 4GB VRAM
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'early_stopping_patience': 8
    }
    
    # Check for glioma model for transfer learning
    glioma_model_path = "../model/best_model.pth"
    if not os.path.exists(glioma_model_path):
        print(f"⚠️ Glioma model not found at {glioma_model_path}")
        glioma_model_path = None
    
    # Initialize trainer
    trainer = SSATrainer(
        device=device,
        patch_dir="SSA_Type/ssa_preprocessed_patches",
        glioma_model_path=glioma_model_path,
        mixed_precision=True
    )
    
    # Start training
    stats = trainer.train(**config)
    
    print(f"\n🎊 SSA BRAIN TUMOR SEGMENTATION TRAINING COMPLETE!")
    print(f"🏆 Best Performance: {stats['best_dice']:.4f} Dice Score")
    print(f"📁 Results saved in: {trainer.save_dir}")

if __name__ == "__main__":
    main()
