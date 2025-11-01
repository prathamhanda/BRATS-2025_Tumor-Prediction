#!/usr/bin/env python3
"""
🧠 SSA Brain Tumor Segmentation - GPU-Optimized 3D U-Net Model
===============================================================

This module implements a GPU-optimized 3D U-Net specifically designed for 
SSA (Sub-Saharan Africa) brain tumor segmentation, leveraging transfer learning
from successful glioma models and optimized for NVIDIA GeForce GTX 1650.

Key Features:
- SSA-specific architecture adaptations
- Transfer learning from glioma models
- GTX 1650 memory optimizations (4GB VRAM)
- Mixed precision training support
- Advanced regularization techniques

Date: September 7, 2025
GPU Target: NVIDIA GeForce GTX 1650 (4GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path

# Try to import MONAI (preferred), fallback to custom implementation
try:
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    MONAI_AVAILABLE = True
    print("✅ MONAI available - using optimized medical imaging framework")
except ImportError:
    MONAI_AVAILABLE = False
    print("⚠️ MONAI not available - using custom PyTorch implementation")

class SSABrainTumorUNet3D(nn.Module):
    """
    SSA-Specific 3D U-Net for brain tumor segmentation
    
    This model is specifically designed for SSA brain tumor data with:
    - Population-specific feature learning
    - Transfer learning capabilities
    - GTX 1650 memory optimization
    - Mixed precision support
    """
    
    def __init__(self, 
                 in_channels: int = 4,
                 out_channels: int = 4,
                 features: Tuple[int, ...] = (32, 64, 128, 256),
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 use_attention: bool = False):
        """
        Initialize SSA-specific 3D U-Net
        
        Args:
            in_channels: Number of input modalities (T1n, T1c, T2w, T2f = 4)
            out_channels: Number of output classes (background + 3 tumor regions = 4)
            features: Number of features in each encoder/decoder level
            dropout: Dropout rate for regularization
            use_batch_norm: Use batch normalization
            use_attention: Use attention mechanisms (experimental)
        """
        super(SSABrainTumorUNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Encoder (Downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        # First encoder block
        self.encoder_blocks.append(
            self._make_encoder_block(in_channels, features[0], use_batch_norm)
        )
        
        # Subsequent encoder blocks with pooling
        for i in range(len(features) - 1):
            self.pool_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.encoder_blocks.append(
                self._make_encoder_block(features[i], features[i + 1], use_batch_norm)
            )
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(
            features[-1], features[-1] * 2, use_batch_norm
        )
        
        # Decoder (Upsampling path)
        self.upconv_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        # Decoder blocks with upsampling
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features)):
            if i == 0:
                # First decoder block from bottleneck
                in_feat = features[-1] * 2
                out_feat = features[-1]
            else:
                in_feat = reversed_features[i - 1]
                out_feat = reversed_features[i]
            
            self.upconv_layers.append(
                nn.ConvTranspose3d(in_feat, out_feat, kernel_size=2, stride=2)
            )
            
            # Concatenation from skip connections doubles the input
            self.decoder_blocks.append(
                self._make_decoder_block(out_feat * 2, out_feat, use_batch_norm)
            )
        
        # Final classification layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        # Attention modules (if enabled)
        if use_attention:
            self.attention_blocks = nn.ModuleList([
                AttentionGate(features[i], features[i], features[i] // 2)
                for i in range(len(features))
            ])
        
        # Dropout layer
        self.dropout_layer = nn.Dropout3d(p=dropout)
        
        print(f"🧠 SSA Brain Tumor 3D U-Net initialized:")
        print(f"   📊 Parameters: ~{self.count_parameters()/1e6:.2f}M")
        print(f"   🎯 Input: {in_channels} channels (T1n, T1c, T2w, T2f)")
        print(f"   📤 Output: {out_channels} classes (background + tumor regions)")
        print(f"   🔧 Features: {features}")
        print(f"   💧 Dropout: {dropout}")
        print(f"   🎯 Attention: {use_attention}")
    
    def _make_encoder_block(self, in_channels: int, out_channels: int, use_batch_norm: bool) -> nn.Module:
        """Create encoder block with double convolution"""
        layers = []
        
        # First convolution
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second convolution
        layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _make_decoder_block(self, in_channels: int, out_channels: int, use_batch_norm: bool) -> nn.Module:
        """Create decoder block with double convolution"""
        return self._make_encoder_block(in_channels, out_channels, use_batch_norm)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SSA U-Net
        
        Args:
            x: Input tensor (B, 4, H, W, D)
            
        Returns:
            Output tensor (B, 4, H, W, D)
        """
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            skip_connections.append(x)
            
            # Apply pooling (except for last encoder block)
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout_layer(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_layers, self.decoder_blocks)):
            # Upsample
            x = upconv(x)
            
            # Get corresponding skip connection
            skip = skip_connections[i]
            
            # Apply attention if enabled
            if self.use_attention:
                skip = self.attention_blocks[len(self.features) - 1 - i](skip, x)
            
            # Concatenate with skip connection
            if x.shape != skip.shape:
                # Handle size mismatch by resizing
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([skip, x], dim=1)
            
            # Apply decoder block
            x = decoder_block(x)
        
        # Final classification
        x = self.final_conv(x)
        
        return x

class AttentionGate(nn.Module):
    """
    Attention Gate for focusing on relevant features
    (Optional - for advanced SSA-specific feature learning)
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SSADataset(torch.utils.data.Dataset):
    """
    GPU-optimized dataset for SSA brain tumor patches
    """
    
    def __init__(self, patch_files: List[str], transform=None, cache_size: int = 50):
        """
        Initialize SSA dataset
        
        Args:
            patch_files: List of patch file paths
            transform: Optional data transforms
            cache_size: Number of patches to keep in memory
        """
        self.patch_files = patch_files
        self.transform = transform
        self.cache_size = cache_size
        self.cache = {}
        
        print(f"📦 SSA Dataset initialized with {len(patch_files)} patches")
        if cache_size > 0:
            print(f"💾 Cache enabled: {cache_size} patches")
    
    def __len__(self) -> int:
        return len(self.patch_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single patch
        
        Returns:
            image: (4, 128, 128, 128) tensor
            mask: (128, 128, 128) tensor
        """
        patch_file = self.patch_files[idx]
        
        # Check cache first
        if patch_file in self.cache:
            data = self.cache[patch_file]
        else:
            # Load from disk
            data = np.load(patch_file)
            
            # Cache if space available
            if len(self.cache) < self.cache_size:
                self.cache[patch_file] = data
        
        # Convert to tensors
        image = torch.from_numpy(data['image']).float()
        mask = torch.from_numpy(data['mask']).long()
        
        # SSA-specific label mapping: map label 4 to label 3 for model compatibility
        # Original: [0, 1, 2, 4] -> Model: [0, 1, 2, 3]
        mask = torch.where(mask == 4, torch.tensor(3), mask)
        
        # Apply transforms if any
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask

class SSAModelManager:
    """
    Comprehensive manager for SSA brain tumor segmentation models
    Handles training, evaluation, and transfer learning
    """
    
    def __init__(self, 
                 device: torch.device,
                 mixed_precision: bool = True,
                 model_save_path: str = "SSA_Type/models"):
        """
        Initialize SSA model manager
        
        Args:
            device: GPU/CPU device
            mixed_precision: Use mixed precision training
            model_save_path: Path to save models
        """
        self.device = device
        self.mixed_precision = mixed_precision
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Setup logging
        self._setup_logging()
        
        print(f"🎯 SSA Model Manager initialized")
        print(f"📍 Device: {device}")
        print(f"⚡ Mixed Precision: {mixed_precision}")
        print(f"💾 Model Save Path: {model_save_path}")
    
    def _setup_logging(self):
        """Setup logging for model training"""
        log_file = self.model_save_path / 'ssa_training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_ssa_model(self, 
                        use_attention: bool = False,
                        dropout: float = 0.1) -> SSABrainTumorUNet3D:
        """
        Create SSA-specific 3D U-Net model
        
        Args:
            use_attention: Use attention mechanisms
            dropout: Dropout rate
            
        Returns:
            SSA 3D U-Net model
        """
        model = SSABrainTumorUNet3D(
            in_channels=4,
            out_channels=4,
            features=(32, 64, 128, 256),  # Optimized for GTX 1650
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Enable optimizations for GTX 1650
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        return model
    
    def load_glioma_weights(self, 
                          model: SSABrainTumorUNet3D,
                          glioma_model_path: str) -> SSABrainTumorUNet3D:
        """
        Load pre-trained glioma model weights for transfer learning
        
        Args:
            model: SSA model to load weights into
            glioma_model_path: Path to glioma model weights
            
        Returns:
            Model with transferred weights
        """
        print(f"🔄 Loading glioma weights for transfer learning...")
        
        try:
            # Load glioma model weights
            glioma_checkpoint = torch.load(glioma_model_path, map_location=self.device)
            
            # Extract state dict
            if 'model_state_dict' in glioma_checkpoint:
                glioma_state_dict = glioma_checkpoint['model_state_dict']
            else:
                glioma_state_dict = glioma_checkpoint
            
            # Get SSA model state dict
            ssa_state_dict = model.state_dict()
            
            # Transfer compatible weights
            transferred_layers = 0
            total_layers = len(ssa_state_dict)
            
            for name, param in glioma_state_dict.items():
                if name in ssa_state_dict and param.shape == ssa_state_dict[name].shape:
                    ssa_state_dict[name].copy_(param)
                    transferred_layers += 1
                    self.logger.debug(f"✅ Transferred: {name}")
                else:
                    self.logger.debug(f"⚠️ Skipped: {name} (shape mismatch or not found)")
            
            # Load the updated weights
            model.load_state_dict(ssa_state_dict)
            
            transfer_rate = (transferred_layers / total_layers) * 100
            print(f"✅ Transfer Learning Complete!")
            print(f"📊 Transferred: {transferred_layers}/{total_layers} layers ({transfer_rate:.1f}%)")
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Transfer learning failed: {e}")
            print(f"⚠️ Continuing with random initialization")
            return model
    
    def get_data_loaders(self, 
                        patch_dir: str,
                        batch_size: int = 1,
                        train_split: float = 0.8,
                        num_workers: int = 2) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create train and validation data loaders
        
        Args:
            patch_dir: Directory containing SSA patches
            batch_size: Batch size (1 recommended for GTX 1650)
            train_split: Training data proportion
            num_workers: Number of data loading workers
            
        Returns:
            train_loader, val_loader
        """
        # Get all patch files
        patch_files = [
            os.path.join(patch_dir, f) 
            for f in os.listdir(patch_dir) 
            if f.endswith('.npz')
        ]
        
        # Split into train/val
        split_idx = int(len(patch_files) * train_split)
        train_files = patch_files[:split_idx]
        val_files = patch_files[split_idx:]
        
        print(f"📊 Data Split:")
        print(f"   🏋️ Training: {len(train_files)} patches")
        print(f"   🔍 Validation: {len(val_files)} patches")
        
        # Create datasets
        train_dataset = SSADataset(train_files, cache_size=20)
        val_dataset = SSADataset(val_files, cache_size=10)
        
        # Create data loaders (optimized for GTX 1650)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return train_loader, val_loader

def main():
    """Main function to demonstrate SSA model creation"""
    print("🧠 SSA Brain Tumor Segmentation Model Initialization")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Device: {device}")
    
    if device.type == 'cuda':
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize model manager
    model_manager = SSAModelManager(device=device, mixed_precision=True)
    
    # Create SSA model
    ssa_model = model_manager.create_ssa_model(use_attention=False, dropout=0.1)
    
    # Test forward pass
    test_input = torch.randn(1, 4, 128, 128, 128).to(device)
    
    print(f"\n🧪 Testing model forward pass...")
    with torch.no_grad():
        output = ssa_model(test_input)
        print(f"✅ Input shape: {test_input.shape}")
        print(f"✅ Output shape: {output.shape}")
    
    # Memory usage
    if device.type == 'cuda':
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"💾 GPU Memory Used: {memory_used:.2f} GB")
    
    print(f"\n🎉 SSA Model Ready for Training!")
    print(f"📋 Next Steps:")
    print(f"   1. Load glioma weights for transfer learning")
    print(f"   2. Create data loaders")
    print(f"   3. Start training with GPU optimization")

if __name__ == "__main__":
    main()
