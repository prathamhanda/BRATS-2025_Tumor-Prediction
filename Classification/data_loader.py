#!/usr/bin/env python3
"""
🧠 Data Loader for Brain Tumor Classification
==============================================

Utility functions to load and prepare data from the existing BraTS datasets
for use with the segmentation-based tumor classifier.

This module handles:
- Loading preprocessed patches from the archive
- Converting between different data formats
- Batch loading for performance testing
- Data validation and quality checks

"""

import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional, Generator
import glob
import random
from dataclasses import dataclass
from enum import Enum
import os

@dataclass
class DatasetInfo:
    """Information about a loaded dataset"""
    name: str
    total_files: int
    glioma_files: int
    ssa_files: int
    patch_shape: Tuple[int, ...]
    modalities: List[str]

class DatasetType(Enum):
    """Types of datasets available"""
    GLIOMA = "glioma"
    SSA = "ssa"
    MIXED = "mixed"

class BraTSDataLoader:
    """
    Data loader for BraTS datasets with support for both original and preprocessed data.
    """
    
    def __init__(self, archive_path: str):
        """
        Initialize the data loader.
        
        Args:
            archive_path: Path to the archive folder containing datasets
        """
        self.archive_path = Path(archive_path)
        self.logger = self._setup_logging()
        
        # Dataset paths
        self.glioma_path = self.archive_path / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        self.ssa_path = self.archive_path / "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
        self.preprocessed_path = self.archive_path / "preprocessed_patches"
        self.met_path = self.archive_path / "MICCAI-LH-BraTS2025-MET-Challenge-TrainingData"
        
        # Check what datasets are available
        self._scan_available_datasets()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _scan_available_datasets(self):
        """Scan and report available datasets"""
        self.logger.info("🔍 Scanning available datasets...")
        
        available_datasets = []
        
        # Check for preprocessed patches
        if self.preprocessed_path.exists():
            patch_files = list(self.preprocessed_path.glob("*.npz"))
            if patch_files:
                available_datasets.append(f"Preprocessed patches: {len(patch_files)} files")
        
        # Check for original datasets
        if self.glioma_path.exists():
            glioma_cases = list(self.glioma_path.glob("BraTS-GLI-*"))
            if glioma_cases:
                available_datasets.append(f"Glioma dataset: {len(glioma_cases)} cases")
        
        if self.ssa_path.exists():
            ssa_cases = list(self.ssa_path.glob("BraTS-SSA-*"))
            if ssa_cases:
                available_datasets.append(f"SSA dataset: {len(ssa_cases)} cases")
        
        if self.met_path.exists():
            met_cases = list(self.met_path.glob("BraTS-MET-*"))
            if met_cases:
                available_datasets.append(f"MET dataset: {len(met_cases)} cases")
        
        if available_datasets:
            self.logger.info("📊 Available datasets:")
            for dataset in available_datasets:
                self.logger.info(f"  • {dataset}")
        else:
            self.logger.warning("⚠️ No datasets found in archive path")
    
    def load_preprocessed_patch(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single preprocessed patch file.
        
        Args:
            file_path: Path to the .npz patch file
            
        Returns:
            Tuple of (image, mask) arrays
        """
        try:
            data = np.load(file_path)
            image = data['image']  # Shape: (4, H, W, D)
            mask = data['mask']    # Shape: (H, W, D)
            
            return image, mask
            
        except Exception as e:
            self.logger.error(f"❌ Error loading patch {file_path}: {e}")
            raise
    
    def load_original_case(self, case_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load original BraTS case with all modalities.
        
        Args:
            case_path: Path to the case directory
            
        Returns:
            Tuple of (image, mask) arrays. Mask may be None if not available.
        """
        case_dir = Path(case_path)
        case_name = case_dir.name
        
        try:
            # Standard BraTS file naming convention
            modalities = ['t1n', 't1c', 't2f', 't2w']  # May vary by dataset
            image_stack = []
            
            # Try to find modality files
            for modality in modalities:
                # Different naming patterns in different datasets
                patterns = [
                    f"{case_name}-{modality}.nii.gz",
                    f"{case_name}_{modality}.nii.gz",
                    f"{case_name}-{modality}.nii",
                ]
                
                modality_file = None
                for pattern in patterns:
                    candidate = case_dir / pattern
                    if candidate.exists():
                        modality_file = candidate
                        break
                
                if modality_file is None:
                    # Try alternative naming
                    nii_files = list(case_dir.glob(f"*{modality}*.nii.gz"))
                    if nii_files:
                        modality_file = nii_files[0]
                
                if modality_file is None:
                    raise ValueError(f"Could not find {modality} modality for {case_name}")
                
                # Load the NIfTI file
                nii_img = nib.load(str(modality_file))
                image_data = nii_img.get_fdata()
                image_stack.append(image_data)
            
            # Stack modalities
            image = np.stack(image_stack, axis=0)  # Shape: (4, H, W, D)
            
            # Try to load segmentation mask
            mask = None
            seg_patterns = [
                f"{case_name}-seg.nii.gz",
                f"{case_name}_seg.nii.gz",
                "seg.nii.gz"
            ]
            
            for pattern in seg_patterns:
                seg_file = case_dir / pattern
                if seg_file.exists():
                    seg_nii = nib.load(str(seg_file))
                    mask = seg_nii.get_fdata()
                    break
            
            return image, mask
            
        except Exception as e:
            self.logger.error(f"❌ Error loading case {case_path}: {e}")
            raise
    
    def get_preprocessed_files(self, 
                              dataset_type: Optional[DatasetType] = None,
                              max_files: Optional[int] = None,
                              shuffle: bool = True) -> List[str]:
        """
        Get list of preprocessed patch files.
        
        Args:
            dataset_type: Filter by dataset type if available
            max_files: Maximum number of files to return
            shuffle: Whether to shuffle the file list
            
        Returns:
            List of file paths
        """
        if not self.preprocessed_path.exists():
            self.logger.warning("⚠️ Preprocessed patches directory not found")
            return []
        
        # Get all .npz files
        patch_files = list(self.preprocessed_path.glob("*.npz"))
        
        if not patch_files:
            self.logger.warning("⚠️ No preprocessed patch files found")
            return []
        
        # Filter by dataset type if specified
        if dataset_type and dataset_type != DatasetType.MIXED:
            # This would require additional logic to identify file sources
            # For now, return all files
            pass
        
        # Convert to strings
        file_paths = [str(f) for f in patch_files]
        
        if shuffle:
            random.shuffle(file_paths)
        
        if max_files:
            file_paths = file_paths[:max_files]
        
        self.logger.info(f"📁 Found {len(file_paths)} preprocessed patch files")
        return file_paths
    
    def get_original_cases(self, 
                          dataset_type: DatasetType,
                          max_cases: Optional[int] = None,
                          shuffle: bool = True) -> List[str]:
        """
        Get list of original case directories.
        
        Args:
            dataset_type: Type of dataset to load
            max_cases: Maximum number of cases to return
            shuffle: Whether to shuffle the case list
            
        Returns:
            List of case directory paths
        """
        if dataset_type == DatasetType.GLIOMA:
            if not self.glioma_path.exists():
                self.logger.warning("⚠️ Glioma dataset path not found")
                return []
            cases = list(self.glioma_path.glob("BraTS-GLI-*"))
            
        elif dataset_type == DatasetType.SSA:
            if not self.ssa_path.exists():
                self.logger.warning("⚠️ SSA dataset path not found")
                return []
            cases = list(self.ssa_path.glob("BraTS-SSA-*"))
            
        else:
            # Mixed dataset
            cases = []
            if self.glioma_path.exists():
                cases.extend(list(self.glioma_path.glob("BraTS-GLI-*")))
            if self.ssa_path.exists():
                cases.extend(list(self.ssa_path.glob("BraTS-SSA-*")))
        
        if not cases:
            self.logger.warning(f"⚠️ No cases found for dataset type: {dataset_type.value}")
            return []
        
        # Convert to strings
        case_paths = [str(case) for case in cases]
        
        if shuffle:
            random.shuffle(case_paths)
        
        if max_cases:
            case_paths = case_paths[:max_cases]
        
        self.logger.info(f"📁 Found {len(case_paths)} {dataset_type.value} cases")
        return case_paths
    
    def create_test_batch(self, 
                         batch_size: int = 5,
                         use_preprocessed: bool = True,
                         dataset_type: DatasetType = DatasetType.MIXED) -> List[Tuple[np.ndarray, str]]:
        """
        Create a test batch for classification experiments.
        
        Args:
            batch_size: Number of samples in the batch
            use_preprocessed: Whether to use preprocessed patches or original data
            dataset_type: Type of dataset to sample from
            
        Returns:
            List of (image, label) tuples where label indicates the dataset source
        """
        batch_data = []
        
        if use_preprocessed:
            # Use preprocessed patches
            patch_files = self.get_preprocessed_files(max_files=batch_size)
            
            for file_path in patch_files:
                try:
                    image, _ = self.load_preprocessed_patch(file_path)
                    
                    # Try to infer label from filename or path
                    if 'glioma' in file_path.lower() or 'gli' in file_path.lower():
                        label = 'glioma'
                    elif 'ssa' in file_path.lower():
                        label = 'ssa'
                    else:
                        label = 'unknown'
                    
                    batch_data.append((image, label))
                    
                except Exception as e:
                    self.logger.error(f"❌ Error loading {file_path}: {e}")
        
        else:
            # Use original cases (this will be slower)
            if dataset_type == DatasetType.MIXED:
                # Mix of both types
                glioma_cases = self.get_original_cases(DatasetType.GLIOMA, max_cases=batch_size//2)
                ssa_cases = self.get_original_cases(DatasetType.SSA, max_cases=batch_size//2)
                
                for case_path in glioma_cases:
                    try:
                        image, _ = self.load_original_case(case_path)
                        batch_data.append((image, 'glioma'))
                    except Exception as e:
                        self.logger.error(f"❌ Error loading {case_path}: {e}")
                
                for case_path in ssa_cases:
                    try:
                        image, _ = self.load_original_case(case_path)
                        batch_data.append((image, 'ssa'))
                    except Exception as e:
                        self.logger.error(f"❌ Error loading {case_path}: {e}")
            
            else:
                # Single dataset type
                cases = self.get_original_cases(dataset_type, max_cases=batch_size)
                label = dataset_type.value
                
                for case_path in cases:
                    try:
                        image, _ = self.load_original_case(case_path)
                        batch_data.append((image, label))
                    except Exception as e:
                        self.logger.error(f"❌ Error loading {case_path}: {e}")
        
        self.logger.info(f"📦 Created test batch with {len(batch_data)} samples")
        return batch_data
    
    def get_dataset_info(self) -> DatasetInfo:
        """
        Get comprehensive information about available datasets.
        
        Returns:
            DatasetInfo object with dataset statistics
        """
        # Count preprocessed files
        preprocessed_files = self.get_preprocessed_files()
        
        # Count original cases
        glioma_cases = self.get_original_cases(DatasetType.GLIOMA)
        ssa_cases = self.get_original_cases(DatasetType.SSA)
        
        # Determine patch shape from a sample file
        patch_shape = None
        if preprocessed_files:
            try:
                sample_image, _ = self.load_preprocessed_patch(preprocessed_files[0])
                patch_shape = sample_image.shape
            except:
                pass
        
        info = DatasetInfo(
            name="BraTS Archive",
            total_files=len(preprocessed_files),
            glioma_files=len(glioma_cases),
            ssa_files=len(ssa_cases),
            patch_shape=patch_shape,
            modalities=['T1', 'T1ce', 'T2', 'FLAIR']
        )
        
        return info
    
    def validate_data_integrity(self, num_samples: int = 10) -> Dict[str, bool]:
        """
        Validate data integrity by checking a random sample of files.
        
        Args:
            num_samples: Number of files to check
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'preprocessed_valid': True,
            'original_glioma_valid': True,
            'original_ssa_valid': True,
            'errors': []
        }
        
        # Check preprocessed files
        patch_files = self.get_preprocessed_files(max_files=num_samples)
        for file_path in patch_files:
            try:
                image, mask = self.load_preprocessed_patch(file_path)
                
                # Basic validation checks
                if image.ndim != 4 or image.shape[0] != 4:
                    results['errors'].append(f"Invalid image shape in {file_path}")
                    results['preprocessed_valid'] = False
                
                if mask is not None and mask.ndim != 3:
                    results['errors'].append(f"Invalid mask shape in {file_path}")
                    results['preprocessed_valid'] = False
                    
            except Exception as e:
                results['errors'].append(f"Error loading {file_path}: {e}")
                results['preprocessed_valid'] = False
        
        # Check original glioma cases
        glioma_cases = self.get_original_cases(DatasetType.GLIOMA, max_cases=num_samples//2)
        for case_path in glioma_cases:
            try:
                image, _ = self.load_original_case(case_path)
                if image.ndim != 4 or image.shape[0] != 4:
                    results['errors'].append(f"Invalid glioma case shape in {case_path}")
                    results['original_glioma_valid'] = False
            except Exception as e:
                results['errors'].append(f"Error loading glioma case {case_path}: {e}")
                results['original_glioma_valid'] = False
        
        # Check original SSA cases
        ssa_cases = self.get_original_cases(DatasetType.SSA, max_cases=num_samples//2)
        for case_path in ssa_cases:
            try:
                image, _ = self.load_original_case(case_path)
                if image.ndim != 4 or image.shape[0] != 4:
                    results['errors'].append(f"Invalid SSA case shape in {case_path}")
                    results['original_ssa_valid'] = False
            except Exception as e:
                results['errors'].append(f"Error loading SSA case {case_path}: {e}")
                results['original_ssa_valid'] = False
        
        return results