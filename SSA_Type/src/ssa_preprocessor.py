#!/usr/bin/env python3
"""
🔄 SSA Brain Tumor Preprocessing Pipeline
========================================

This module implements a comprehensive preprocessing pipeline for SSA (Sub-Saharan Africa) 
brain tumor data, adapted from the successful glioma preprocessing approach with 
SSA-specific optimizations.

Key Features:
- Label mapping (SSA label 3 → standard label 4)
- Quality filtering and validation
- Patch extraction (128³ voxels)
- Intensity normalization
- Population-aware augmentations

Author: Research Team
Date: September 2025
Purpose: SSA tumor segmentation research
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
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SSAPreprocessor:
    """
    Comprehensive SSA brain tumor data preprocessing pipeline
    
    This class handles the complete preprocessing workflow for SSA brain tumor data,
    including quality assessment, label mapping, patch extraction, and normalization.
    """
    
    def __init__(self, 
                 ssa_data_path: str,
                 output_path: str = "SSA_Type/ssa_preprocessed_patches",
                 patch_size: int = 128,
                 enable_quality_filter: bool = True):
        """
        Initialize the SSA preprocessor
        
        Args:
            ssa_data_path: Path to SSA raw dataset
            output_path: Path for preprocessed patch output
            patch_size: Size of extracted patches (default: 128)
            enable_quality_filter: Whether to apply quality filtering
        """
        self.ssa_data_path = Path(ssa_data_path)
        self.output_path = Path(output_path)
        self.patch_size = patch_size
        self.enable_quality_filter = enable_quality_filter
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing statistics
        self.preprocessing_stats = {
            'total_cases': 0,
            'processed_cases': 0,
            'failed_cases': 0,
            'total_patches': 0,
            'quality_filtered': 0,
            'label_mapping_applied': 0,
            'processing_errors': []
        }
        
        # SSA-specific configuration
        self.modalities = ['t1n', 't1c', 't2w', 't2f']
        self.target_shape = (240, 240, 155)
        self.target_spacing = (1.0, 1.0, 1.0)
        
        print("🔄 SSA Preprocessor Initialized")
        print("=" * 40)
        print(f"📁 Input Path: {self.ssa_data_path}")
        print(f"📦 Output Path: {self.output_path}")
        print(f"📐 Patch Size: {self.patch_size}³ voxels")
        print(f"🔍 Quality Filter: {'Enabled' if self.enable_quality_filter else 'Disabled'}")
        
    def discover_ssa_cases(self) -> List[str]:
        """
        Discover all available SSA cases
        
        Returns:
            List of SSA case IDs
        """
        print("\n🔍 DISCOVERING SSA CASES...")
        print("-" * 30)
        
        case_dirs = []
        if self.ssa_data_path.exists():
            for item in self.ssa_data_path.iterdir():
                if item.is_dir() and 'BraTS-SSA-' in item.name:
                    case_dirs.append(item.name)
        
        case_dirs = sorted(case_dirs)
        self.preprocessing_stats['total_cases'] = len(case_dirs)
        
        print(f"✅ Found {len(case_dirs)} SSA cases")
        if case_dirs:
            print(f"📋 Range: {case_dirs[0]} → {case_dirs[-1]}")
        
        return case_dirs
    
    def validate_case_quality(self, case_path: Path, case_id: str) -> Tuple[bool, Dict]:
        """
        Validate the quality of a single SSA case
        
        Args:
            case_path: Path to the case directory
            case_id: Case identifier
            
        Returns:
            Tuple of (is_valid, quality_info)
        """
        quality_info = {
            'case_id': case_id,
            'files_present': [],
            'missing_files': [],
            'file_sizes': {},
            'shape_consistency': True,
            'intensity_ranges': {},
            'is_valid': True,
            'issues': []
        }
        
        # Check file presence
        required_files = self.modalities + ['seg']
        for modality in required_files:
            file_path = case_path / f"{case_id}-{modality}.nii.gz"
            if file_path.exists():
                quality_info['files_present'].append(modality)
                # Get file size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                quality_info['file_sizes'][modality] = round(size_mb, 2)
            else:
                quality_info['missing_files'].append(modality)
                quality_info['issues'].append(f"Missing {modality} file")
        
        # If any files are missing, mark as invalid
        if quality_info['missing_files']:
            quality_info['is_valid'] = False
            return False, quality_info
        
        # Check image properties for first modality
        try:
            first_modality_path = case_path / f"{case_id}-{self.modalities[0]}.nii.gz"
            nii_img = nib.load(first_modality_path)
            img_data = nii_img.get_fdata()
            
            # Check dimensions
            if img_data.shape != self.target_shape:
                quality_info['issues'].append(f"Unexpected shape: {img_data.shape}")
                quality_info['shape_consistency'] = False
                
            # Check voxel spacing
            spacing = nii_img.header.get_zooms()[:3]
            if not np.allclose(spacing, self.target_spacing, atol=0.1):
                quality_info['issues'].append(f"Unexpected spacing: {spacing}")
            
            # Check intensity range (for non-zero regions)
            brain_mask = img_data > 0
            if np.any(brain_mask):
                intensity_stats = {
                    'min': float(np.min(img_data[brain_mask])),
                    'max': float(np.max(img_data[brain_mask])),
                    'mean': float(np.mean(img_data[brain_mask])),
                    'std': float(np.std(img_data[brain_mask]))
                }
                quality_info['intensity_ranges'][self.modalities[0]] = intensity_stats
                
        except Exception as e:
            quality_info['issues'].append(f"Error loading {self.modalities[0]}: {str(e)}")
            quality_info['is_valid'] = False
        
        # Apply quality filtering rules
        if self.enable_quality_filter:
            # Rule 1: File size check (too small might indicate corruption)
            for modality in self.modalities:
                if modality in quality_info['file_sizes']:
                    if quality_info['file_sizes'][modality] < 1.0:  # Less than 1MB
                        quality_info['issues'].append(f"{modality} file too small ({quality_info['file_sizes'][modality]} MB)")
                        quality_info['is_valid'] = False
            
            # Rule 2: Shape consistency
            if not quality_info['shape_consistency']:
                quality_info['is_valid'] = False
        
        return quality_info['is_valid'], quality_info
    
    def apply_label_mapping(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply SSA-specific label mapping to match standard BraTS format
        
        SSA Format: 0 (background), 1 (NCR/NET), 2 (edema), 3 (enhancing)
        Standard Format: 0 (background), 1 (NCR/NET), 2 (edema), 4 (enhancing)
        
        Args:
            segmentation: Original segmentation array
            
        Returns:
            Mapped segmentation array
        """
        mapped_seg = segmentation.copy()
        
        # Map label 3 to label 4 for enhancing tumor
        mapped_seg[segmentation == 3] = 4
        
        return mapped_seg
    
    def normalize_modality(self, image: np.ndarray, modality: str) -> np.ndarray:
        """
        Apply modality-specific normalization
        
        Args:
            image: Input image array
            modality: Modality name (t1n, t1c, t2w, t2f)
            
        Returns:
            Normalized image array
        """
        normalized_image = image.copy()
        
        # Create brain mask (non-zero regions)
        brain_mask = image > 0
        
        if np.any(brain_mask):
            # Z-score normalization within brain region
            brain_values = image[brain_mask]
            mean_val = np.mean(brain_values)
            std_val = np.std(brain_values)
            
            if std_val > 0:
                normalized_image[brain_mask] = (brain_values - mean_val) / std_val
            
            # Clip extreme values (beyond 3 standard deviations)
            normalized_image = np.clip(normalized_image, -3, 3)
        
        return normalized_image.astype(np.float32)
    
    def extract_patches(self, 
                       image: np.ndarray, 
                       segmentation: np.ndarray,
                       case_id: str,
                       overlap_threshold: int = 100) -> List[Dict]:
        """
        Extract patches from normalized image and segmentation
        
        Args:
            image: Multi-modal image array (4, H, W, D)
            segmentation: Segmentation array (H, W, D)
            case_id: Case identifier
            overlap_threshold: Minimum tumor voxels to include patch
            
        Returns:
            List of patch dictionaries
        """
        patches = []
        h, w, d = image.shape[1:]  # Get spatial dimensions
        
        # Calculate patch extraction coordinates
        stride = self.patch_size // 2  # 50% overlap
        
        patch_idx = 0
        for z in range(0, max(1, d - self.patch_size + 1), stride):
            for y in range(0, max(1, h - self.patch_size + 1), stride):
                for x in range(0, max(1, w - self.patch_size + 1), stride):
                    
                    # Extract patch bounds
                    z_end = min(z + self.patch_size, d)
                    y_end = min(y + self.patch_size, h)
                    x_end = min(x + self.patch_size, w)
                    
                    # Extract patches
                    image_patch = image[:, y:y_end, x:x_end, z:z_end]
                    seg_patch = segmentation[y:y_end, x:x_end, z:z_end]
                    
                    # Pad if necessary to reach target size
                    if image_patch.shape != (4, self.patch_size, self.patch_size, self.patch_size):
                        padded_image = np.zeros((4, self.patch_size, self.patch_size, self.patch_size), dtype=np.float32)
                        padded_seg = np.zeros((self.patch_size, self.patch_size, self.patch_size), dtype=np.uint8)
                        
                        # Copy actual data
                        padded_image[:, :image_patch.shape[1], :image_patch.shape[2], :image_patch.shape[3]] = image_patch
                        padded_seg[:seg_patch.shape[0], :seg_patch.shape[1], :seg_patch.shape[2]] = seg_patch
                        
                        image_patch = padded_image
                        seg_patch = padded_seg
                    
                    # Quality check: only save patches with sufficient content
                    tumor_voxels = np.sum(seg_patch > 0)
                    brain_voxels = np.sum(image_patch[0] > 0)  # Use T1n as reference
                    
                    if tumor_voxels >= overlap_threshold or brain_voxels >= self.patch_size**3 * 0.1:
                        patch_info = {
                            'image': image_patch.astype(np.float32),
                            'mask': seg_patch.astype(np.uint8),
                            'case_id': case_id,
                            'patch_idx': patch_idx,
                            'coordinates': (x, y, z),
                            'tumor_voxels': int(tumor_voxels),
                            'brain_voxels': int(brain_voxels)
                        }
                        patches.append(patch_info)
                        patch_idx += 1
        
        return patches
    
    def process_single_case(self, case_id: str) -> Dict:
        """
        Process a single SSA case through the complete pipeline
        
        Args:
            case_id: Case identifier
            
        Returns:
            Processing results dictionary
        """
        case_path = self.ssa_data_path / case_id
        
        processing_result = {
            'case_id': case_id,
            'success': False,
            'patches_created': 0,
            'quality_info': {},
            'errors': []
        }
        
        try:
            print(f"🔄 Processing: {case_id}")
            
            # Step 1: Quality validation
            is_valid, quality_info = self.validate_case_quality(case_path, case_id)
            processing_result['quality_info'] = quality_info
            
            if not is_valid:
                processing_result['errors'].append("Quality validation failed")
                self.preprocessing_stats['quality_filtered'] += 1
                return processing_result
            
            # Step 2: Load all modalities
            images = []
            for modality in self.modalities:
                file_path = case_path / f"{case_id}-{modality}.nii.gz"
                nii_img = nib.load(file_path)
                img_data = nii_img.get_fdata()
                
                # Normalize modality
                normalized_img = self.normalize_modality(img_data, modality)
                images.append(normalized_img)
            
            # Stack into multi-channel image (4, H, W, D)
            multi_modal_image = np.stack(images, axis=0)
            
            # Step 3: Load and process segmentation
            seg_path = case_path / f"{case_id}-seg.nii.gz"
            seg_nii = nib.load(seg_path)
            segmentation = seg_nii.get_fdata()
            
            # Apply SSA label mapping
            mapped_segmentation = self.apply_label_mapping(segmentation)
            self.preprocessing_stats['label_mapping_applied'] += 1
            
            # Step 4: Extract patches
            patches = self.extract_patches(multi_modal_image, mapped_segmentation, case_id)
            
            # Step 5: Save patches
            patches_saved = 0
            for patch in patches:
                patch_filename = f"{case_id}_patch_{patch['patch_idx']:03d}.npz"
                patch_path = self.output_path / patch_filename
                
                # Save patch data
                np.savez_compressed(
                    patch_path,
                    image=patch['image'],
                    mask=patch['mask'],
                    case_id=case_id,
                    patch_idx=patch['patch_idx'],
                    coordinates=patch['coordinates'],
                    tumor_voxels=patch['tumor_voxels'],
                    brain_voxels=patch['brain_voxels']
                )
                patches_saved += 1
            
            processing_result['success'] = True
            processing_result['patches_created'] = patches_saved
            self.preprocessing_stats['processed_cases'] += 1
            self.preprocessing_stats['total_patches'] += patches_saved
            
            print(f"   ✅ Success: {patches_saved} patches created")
            
        except Exception as e:
            error_msg = f"Error processing {case_id}: {str(e)}"
            processing_result['errors'].append(error_msg)
            self.preprocessing_stats['failed_cases'] += 1
            self.preprocessing_stats['processing_errors'].append(error_msg)
            print(f"   ❌ Error: {str(e)}")
        
        return processing_result
    
    def create_preprocessing_summary(self, processing_results: List[Dict]) -> Dict:
        """
        Create a comprehensive summary of preprocessing results
        
        Args:
            processing_results: List of processing results from all cases
            
        Returns:
            Summary dictionary
        """
        summary = {
            'preprocessing_date': datetime.now().isoformat(),
            'configuration': {
                'patch_size': self.patch_size,
                'quality_filter_enabled': self.enable_quality_filter,
                'modalities': self.modalities,
                'label_mapping': 'SSA label 3 → standard label 4'
            },
            'statistics': self.preprocessing_stats,
            'successful_cases': [],
            'failed_cases': [],
            'quality_filtered_cases': []
        }
        
        # Categorize results
        for result in processing_results:
            if result['success']:
                summary['successful_cases'].append({
                    'case_id': result['case_id'],
                    'patches_created': result['patches_created']
                })
            else:
                if result['quality_info'].get('is_valid', True) == False:
                    summary['quality_filtered_cases'].append({
                        'case_id': result['case_id'],
                        'issues': result['quality_info'].get('issues', [])
                    })
                else:
                    summary['failed_cases'].append({
                        'case_id': result['case_id'],
                        'errors': result['errors']
                    })
        
        # Calculate success rate
        total_attempted = len(processing_results)
        if total_attempted > 0:
            summary['statistics']['success_rate'] = (summary['statistics']['processed_cases'] / total_attempted) * 100
        
        # Save summary
        summary_path = self.output_path / "ssa_preprocessing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def run_complete_preprocessing(self, max_cases: Optional[int] = None) -> Dict:
        """
        Run the complete SSA preprocessing pipeline
        
        Args:
            max_cases: Maximum number of cases to process (None for all)
            
        Returns:
            Complete preprocessing summary
        """
        print("🚀 STARTING COMPLETE SSA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Discover cases
        case_list = self.discover_ssa_cases()
        
        if max_cases:
            case_list = case_list[:max_cases]
            print(f"🎯 Processing limited to {max_cases} cases")
        
        if not case_list:
            print("❌ No SSA cases found for processing!")
            return {}
        
        print(f"📊 Total cases to process: {len(case_list)}")
        print("-" * 40)
        
        # Process all cases
        processing_results = []
        for case_id in tqdm(case_list, desc="Processing SSA cases"):
            result = self.process_single_case(case_id)
            processing_results.append(result)
        
        # Create summary
        summary = self.create_preprocessing_summary(processing_results)
        
        # Print results
        print("\n🎉 SSA PREPROCESSING COMPLETE!")
        print("=" * 45)
        print(f"📊 Statistics:")
        print(f"   - Total Cases: {summary['statistics']['total_cases']}")
        print(f"   - Processed Successfully: {summary['statistics']['processed_cases']}")
        print(f"   - Quality Filtered: {summary['statistics']['quality_filtered']}")
        print(f"   - Failed: {summary['statistics']['failed_cases']}")
        print(f"   - Total Patches Created: {summary['statistics']['total_patches']}")
        print(f"   - Success Rate: {summary['statistics'].get('success_rate', 0):.1f}%")
        print(f"\n📁 Generated Files:")
        print(f"   - Preprocessed patches: {self.output_path}")
        print(f"   - Summary report: ssa_preprocessing_summary.json")
        
        return summary


def main():
    """
    Main function to run SSA preprocessing
    """
    # Configuration
    ssa_dataset_path = "f:/Projects/BrainTumorDetector/archive/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
    output_path = "SSA_Type/ssa_preprocessed_patches"
    
    # Initialize preprocessor
    preprocessor = SSAPreprocessor(
        ssa_data_path=ssa_dataset_path,
        output_path=output_path,
        patch_size=128,
        enable_quality_filter=True
    )
    
    # Run preprocessing (process first 10 cases for demonstration)
    summary = preprocessor.run_complete_preprocessing(max_cases=10)
    
    return summary


if __name__ == "__main__":
    summary = main()
