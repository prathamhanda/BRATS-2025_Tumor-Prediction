#!/usr/bin/env python3
"""
🚀 GPU-Optimized SSA Brain Tumor Preprocessing Pipeline
=======================================================

This module implements a GPU-accelerated preprocessing pipeline for SSA brain tumor data.
Optimized for NVIDIA GeForce GTX 1650 with 4GB VRAM.

Key GPU Optimizations:
- CUDA tensor operations for intensive computations
- Memory-efficient batch processing
- Mixed precision support
- Optimized data loading and normalization
- Smart memory management for GTX 1650

Date: September 7, 2025
GPU Target: NVIDIA GeForce GTX 1650 (4GB VRAM)
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import glob
from tqdm import tqdm
import psutil
import gc
import logging

# GPU acceleration imports
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GPUOptimizedSSAPreprocessor:
    """
    GPU-Accelerated SSA brain tumor data preprocessing pipeline
    
    This class leverages GPU acceleration for intensive preprocessing tasks
    while maintaining compatibility with limited VRAM (4GB GTX 1650).
    """
    
    def __init__(self, 
                 ssa_data_path: str,
                 output_path: str = "SSA_Type/ssa_preprocessed_patches",
                 patch_size: int = 128,
                 enable_gpu: bool = True,
                 mixed_precision: bool = True,
                 batch_size: int = 1):
        """
        Initialize GPU-optimized SSA preprocessor
        
        Args:
            ssa_data_path: Path to SSA dataset
            output_path: Output directory for processed patches
            patch_size: Size of extracted patches (128 recommended for GTX 1650)
            enable_gpu: Use GPU acceleration if available
            mixed_precision: Use mixed precision for memory efficiency
            batch_size: Processing batch size (1 recommended for GTX 1650)
        """
        self.ssa_data_path = Path(ssa_data_path)
        self.output_path = Path(output_path)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        
        # GPU setup
        self.device = self._setup_gpu(enable_gpu)
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_cases': 0,
            'processed_cases': 0,
            'total_patches': 0,
            'failed_cases': 0,
            'gpu_memory_peak': 0,
            'processing_times': []
        }
        
        # Setup logging
        self._setup_logging()
        
        print(f"🚀 GPU-Optimized SSA Preprocessor Initialized")
        print(f"📍 Device: {self.device}")
        print(f"💾 Mixed Precision: {self.mixed_precision}")
        print(f"🔢 Batch Size: {batch_size}")
        print(f"📦 Patch Size: {patch_size}³ voxels")
        
        if self.enable_gpu:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎯 GPU Memory: {gpu_memory:.1f} GB available")
    
    def _setup_gpu(self, enable_gpu: bool) -> torch.device:
        """Setup GPU device and optimizations"""
        if enable_gpu and torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # GPU optimizations for GTX 1650
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            print(f"✅ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print("⚠️ GPU not available, using CPU")
            return torch.device('cpu')
    
    def _setup_logging(self):
        """Setup logging for preprocessing operations"""
        log_file = self.output_path / 'gpu_preprocessing.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def monitor_gpu_memory(self) -> Dict[str, float]:
        """Monitor GPU memory usage"""
        if not self.enable_gpu:
            return {'allocated': 0, 'reserved': 0, 'free': 0}
            
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - reserved
        
        # Update peak memory usage
        self.stats['gpu_memory_peak'] = max(self.stats['gpu_memory_peak'], reserved)
        
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'free': free,
            'total': total
        }
    
    def gpu_normalize_intensity(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated intensity normalization
        
        Args:
            image_tensor: Input image tensor (C, H, W, D)
            
        Returns:
            Normalized image tensor
        """
        with autocast(enabled=self.mixed_precision):
            normalized = torch.zeros_like(image_tensor)
            
            for i in range(image_tensor.shape[0]):  # For each modality
                modality = image_tensor[i]
                
                # Create brain mask (non-zero voxels)
                brain_mask = modality > 0
                
                if brain_mask.sum() > 0:
                    # Calculate mean and std only for brain voxels
                    brain_voxels = modality[brain_mask]
                    mean_val = brain_voxels.mean()
                    std_val = brain_voxels.std()
                    
                    if std_val > 0:
                        # Normalize: (x - mean) / std
                        normalized[i] = (modality - mean_val) / std_val
                        # Zero out background
                        normalized[i] = normalized[i] * brain_mask.float()
                    else:
                        normalized[i] = modality
                else:
                    normalized[i] = modality
                    
        return normalized
    
    def gpu_extract_patches(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> List[Dict]:
        """
        GPU-accelerated patch extraction with intelligent selection
        
        Args:
            image_tensor: Image tensor (C, H, W, D)
            mask_tensor: Mask tensor (H, W, D)
            
        Returns:
            List of extracted patches with metadata
        """
        patches = []
        h, w, d = image_tensor.shape[1:]
        patch_size = self.patch_size
        
        # Calculate stride for patch extraction (50% overlap)
        stride = patch_size // 2
        
        with autocast(enabled=self.mixed_precision):
            for z in range(0, max(1, d - patch_size + 1), stride):
                for y in range(0, max(1, h - patch_size + 1), stride):
                    for x in range(0, max(1, w - patch_size + 1), stride):
                        
                        # Extract patch coordinates
                        z_end = min(z + patch_size, d)
                        y_end = min(y + patch_size, h)
                        x_end = min(x + patch_size, w)
                        
                        # Extract patches
                        image_patch = image_tensor[:, y:y_end, x:x_end, z:z_end]
                        mask_patch = mask_tensor[y:y_end, x:x_end, z:z_end]
                        
                        # Pad if necessary
                        if image_patch.shape != (4, patch_size, patch_size, patch_size):
                            padded_image = torch.zeros(4, patch_size, patch_size, patch_size, 
                                                     device=self.device, dtype=image_patch.dtype)
                            padded_mask = torch.zeros(patch_size, patch_size, patch_size,
                                                    device=self.device, dtype=mask_patch.dtype)
                            
                            # Copy actual data
                            padded_image[:, :image_patch.shape[1], :image_patch.shape[2], :image_patch.shape[3]] = image_patch
                            padded_mask[:mask_patch.shape[0], :mask_patch.shape[1], :mask_patch.shape[2]] = mask_patch
                            
                            image_patch = padded_image
                            mask_patch = padded_mask
                        
                        # Quality check: only save patches with sufficient tumor content
                        tumor_voxels = (mask_patch > 0).sum().item()
                        
                        if tumor_voxels >= 50:  # Minimum tumor voxels threshold
                            patches.append({
                                'image': image_patch.cpu().numpy(),
                                'mask': mask_patch.cpu().numpy(),
                                'tumor_voxels': tumor_voxels,
                                'coordinates': (x, y, z)
                            })
                            
                            # Memory management for GTX 1650
                            if len(patches) % 10 == 0:
                                torch.cuda.empty_cache()
        
        return patches
    
    def process_single_case(self, case_path: Path) -> int:
        """
        Process a single SSA case with GPU acceleration
        
        Args:
            case_path: Path to the case directory
            
        Returns:
            Number of patches created
        """
        case_name = case_path.name
        self.logger.info(f"🔄 Processing {case_name} with GPU acceleration...")
        
        try:
            start_time = datetime.now()
            
            # Find required files
            files = {
                't1n': glob.glob(str(case_path / "*t1n.nii.gz")),
                't1c': glob.glob(str(case_path / "*t1c.nii.gz")),
                't2w': glob.glob(str(case_path / "*t2w.nii.gz")),
                't2f': glob.glob(str(case_path / "*t2f.nii.gz")),
                'seg': glob.glob(str(case_path / "*seg.nii.gz"))
            }
            
            # Validate files
            missing_files = [k for k, v in files.items() if not v]
            if missing_files:
                self.logger.warning(f"⚠️ Missing files in {case_name}: {missing_files}")
                return 0
            
            # Load data and convert to GPU tensors
            self.logger.info(f"📂 Loading data for {case_name}...")
            
            # Load MRI modalities
            t1n = torch.from_numpy(nib.load(files['t1n'][0]).get_fdata()).float().to(self.device)
            t1c = torch.from_numpy(nib.load(files['t1c'][0]).get_fdata()).float().to(self.device)
            t2w = torch.from_numpy(nib.load(files['t2w'][0]).get_fdata()).float().to(self.device)
            t2f = torch.from_numpy(nib.load(files['t2f'][0]).get_fdata()).float().to(self.device)
            seg = torch.from_numpy(nib.load(files['seg'][0]).get_fdata()).long().to(self.device)
            
            # Stack modalities
            image = torch.stack([t1n, t1c, t2w, t2f], dim=0)
            
            # SSA-specific label mapping (label 3 → label 4)
            seg = torch.where(seg == 3, torch.tensor(4, device=self.device), seg)
            
            # GPU-accelerated intensity normalization
            self.logger.info(f"🔄 GPU normalization for {case_name}...")
            image = self.gpu_normalize_intensity(image)
            
            # Monitor memory usage
            memory_info = self.monitor_gpu_memory()
            self.logger.info(f"💾 GPU Memory: {memory_info['reserved']:.2f}GB used, {memory_info['free']:.2f}GB free")
            
            # Extract patches with GPU acceleration
            self.logger.info(f"✂️ GPU patch extraction for {case_name}...")
            patches = self.gpu_extract_patches(image, seg)
            
            # Save patches
            patches_saved = 0
            for i, patch_data in enumerate(patches):
                patch_filename = f"{case_name}_patch_{i:03d}.npz"
                patch_path = self.output_path / patch_filename
                
                np.savez_compressed(
                    patch_path,
                    image=patch_data['image'].astype(np.float32),
                    mask=patch_data['mask'].astype(np.uint8),
                    tumor_voxels=patch_data['tumor_voxels'],
                    coordinates=patch_data['coordinates'],
                    case_name=case_name
                )
                patches_saved += 1
            
            # Cleanup GPU memory
            del image, seg, t1n, t1c, t2w, t2f, patches
            torch.cuda.empty_cache()
            
            # Record processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_times'].append(processing_time)
            
            self.logger.info(f"✅ {case_name}: {patches_saved} patches created in {processing_time:.1f}s")
            return patches_saved
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {case_name}: {str(e)}")
            self.stats['failed_cases'] += 1
            
            # Cleanup on error
            torch.cuda.empty_cache()
            return 0
    
    def process_dataset(self, max_cases: Optional[int] = None) -> Dict:
        """
        Process the complete SSA dataset with GPU acceleration
        
        Args:
            max_cases: Maximum number of cases to process (None for all)
            
        Returns:
            Processing statistics
        """
        print(f"\n🚀 Starting GPU-Accelerated SSA Dataset Processing")
        print("=" * 70)
        
        # Find all case directories
        case_dirs = [d for d in self.ssa_data_path.iterdir() if d.is_dir() and 'BraTS-SSA' in d.name]
        
        if max_cases:
            case_dirs = case_dirs[:max_cases]
            
        self.stats['total_cases'] = len(case_dirs)
        
        print(f"📊 Found {len(case_dirs)} SSA cases to process")
        print(f"🎯 GPU Device: {self.device}")
        print(f"💾 Mixed Precision: {self.mixed_precision}")
        
        # Process cases with progress bar
        total_patches = 0
        
        with tqdm(case_dirs, desc="🧠 Processing SSA Cases") as pbar:
            for case_dir in pbar:
                patches_created = self.process_single_case(case_dir)
                total_patches += patches_created
                self.stats['processed_cases'] += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Patches': total_patches,
                    'GPU_Mem': f"{self.monitor_gpu_memory()['reserved']:.1f}GB",
                    'Cases': f"{self.stats['processed_cases']}/{self.stats['total_cases']}"
                })
        
        self.stats['total_patches'] = total_patches
        
        # Generate final report
        self._generate_processing_report()
        
        return self.stats
    
    def _generate_processing_report(self):
        """Generate comprehensive processing report"""
        print(f"\n📊 GPU-ACCELERATED SSA PREPROCESSING COMPLETE")
        print("=" * 70)
        
        # Processing statistics
        success_rate = (self.stats['processed_cases'] / self.stats['total_cases']) * 100
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        print(f"✅ Cases Processed: {self.stats['processed_cases']}/{self.stats['total_cases']} ({success_rate:.1f}%)")
        print(f"📦 Total Patches Created: {self.stats['total_patches']}")
        print(f"❌ Failed Cases: {self.stats['failed_cases']}")
        print(f"⏱️ Average Processing Time: {avg_time:.1f}s per case")
        print(f"🎯 Peak GPU Memory Usage: {self.stats['gpu_memory_peak']:.2f} GB")
        
        # GPU utilization summary
        if self.enable_gpu:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            utilization = (self.stats['gpu_memory_peak'] / gpu_memory) * 100
            print(f"📊 GPU Utilization: {utilization:.1f}% of {gpu_memory:.1f}GB")
        
        # Save detailed statistics
        stats_file = self.output_path / 'gpu_preprocessing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        print(f"💾 Detailed statistics saved to: {stats_file}")
        print(f"📁 Processed patches saved to: {self.output_path}")

def main():
    """Main execution function"""
    print("🚀 GPU-Optimized SSA Brain Tumor Preprocessing")
    print("=" * 70)
    
    # Configuration for GTX 1650 optimization
    config = {
        'ssa_data_path': "../archive/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2",
        'output_path': "SSA_Type/ssa_preprocessed_patches", 
        'patch_size': 128,  # Optimal for GTX 1650
        'enable_gpu': True,
        'mixed_precision': True,  # Essential for memory efficiency
        'batch_size': 1  # Conservative for 4GB VRAM
    }
    
    # Check if data exists
    data_path = Path(config['ssa_data_path'])
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print("Please ensure SSA dataset is available in the specified location")
        return
    
    # Initialize GPU-optimized preprocessor
    preprocessor = GPUOptimizedSSAPreprocessor(**config)
    
    # Process dataset
    stats = preprocessor.process_dataset(max_cases=5)  # Start with 5 cases for testing
    
    print(f"\n🎉 GPU-ACCELERATED PREPROCESSING COMPLETE!")
    print(f"🧠 Ready for SSA brain tumor segmentation training")

if __name__ == "__main__":
    main()
