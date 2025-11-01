#!/usr/bin/env python3
"""
🧠 SSA Brain Tumor Dataset Explorer
==================================

This module provides comprehensive analysis of the SSA (Sub-Saharan Africa) brain tumor dataset
from the BraTS challenge. It explores data characteristics, quality, and population-specific patterns.

Author: Research Team
Date: September 2025
Purpose: SSA tumor segmentation research
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SSADatasetExplorer:
    """
    Comprehensive SSA dataset analysis tool
    
    This class provides methods to analyze the SSA brain tumor dataset,
    including data quality assessment, population-specific characteristics,
    and comparative analysis with standard glioma datasets.
    """
    
    def __init__(self, ssa_data_path: str):
        """
        Initialize the SSA dataset explorer
        
        Args:
            ssa_data_path: Path to SSA dataset directory
        """
        self.ssa_data_path = Path(ssa_data_path)
        self.analysis_results = {}
        self.case_list = []
        self.modalities = ['t1n', 't1c', 't2w', 't2f', 'seg']
        
        print("🧠 SSA Dataset Explorer Initialized")
        print("=" * 50)
        print(f"📁 Dataset Path: {self.ssa_data_path}")
        
    def discover_cases(self) -> List[str]:
        """
        Discover all SSA cases in the dataset
        
        Returns:
            List of case IDs found in the dataset
        """
        print("\n🔍 DISCOVERING SSA CASES...")
        print("-" * 30)
        
        case_dirs = []
        if self.ssa_data_path.exists():
            for item in self.ssa_data_path.iterdir():
                if item.is_dir() and 'BraTS-SSA-' in item.name:
                    case_dirs.append(item.name)
        
        self.case_list = sorted(case_dirs)
        
        print(f"✅ Found {len(self.case_list)} SSA cases")
        if self.case_list:
            print(f"📋 Case Range: {self.case_list[0]} → {self.case_list[-1]}")
            print(f"🎯 Sample Cases: {', '.join(self.case_list[:3])}...")
        
        return self.case_list
    
    def analyze_case_structure(self) -> Dict:
        """
        Analyze the file structure of SSA cases
        
        Returns:
            Dictionary containing structure analysis results
        """
        print("\n📁 ANALYZING CASE STRUCTURE...")
        print("-" * 35)
        
        structure_analysis = {
            'complete_cases': 0,
            'incomplete_cases': 0,
            'missing_modalities': {},
            'file_sizes': {},
            'case_details': []
        }
        
        for case_id in self.case_list[:10]:  # Analyze first 10 cases
            case_path = self.ssa_data_path / case_id
            case_info = {
                'case_id': case_id,
                'files_found': [],
                'missing_files': [],
                'file_sizes': {}
            }
            
            # Check for each modality
            files_present = 0
            for modality in self.modalities:
                expected_file = f"{case_id}-{modality}.nii.gz"
                file_path = case_path / expected_file
                
                if file_path.exists():
                    case_info['files_found'].append(modality)
                    # Get file size in MB
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    case_info['file_sizes'][modality] = round(size_mb, 2)
                    files_present += 1
                else:
                    case_info['missing_files'].append(modality)
                    # Track missing modalities globally
                    if modality not in structure_analysis['missing_modalities']:
                        structure_analysis['missing_modalities'][modality] = 0
                    structure_analysis['missing_modalities'][modality] += 1
            
            # Classify case completeness
            if files_present == len(self.modalities):
                structure_analysis['complete_cases'] += 1
            else:
                structure_analysis['incomplete_cases'] += 1
            
            structure_analysis['case_details'].append(case_info)
            
            print(f"📄 {case_id}: {files_present}/{len(self.modalities)} files present")
        
        # Calculate average file sizes
        for modality in self.modalities:
            sizes = [case['file_sizes'].get(modality, 0) for case in structure_analysis['case_details']]
            valid_sizes = [s for s in sizes if s > 0]
            if valid_sizes:
                structure_analysis['file_sizes'][modality] = {
                    'mean': round(np.mean(valid_sizes), 2),
                    'std': round(np.std(valid_sizes), 2),
                    'range': f"{min(valid_sizes):.1f}-{max(valid_sizes):.1f} MB"
                }
        
        self.analysis_results['structure'] = structure_analysis
        
        print(f"\n✅ Structure Analysis Complete:")
        print(f"   📊 Complete Cases: {structure_analysis['complete_cases']}")
        print(f"   ⚠️ Incomplete Cases: {structure_analysis['incomplete_cases']}")
        
        return structure_analysis
    
    def analyze_image_properties(self, max_cases: int = 5) -> Dict:
        """
        Analyze image properties (dimensions, spacing, orientation)
        
        Args:
            max_cases: Maximum number of cases to analyze for performance
            
        Returns:
            Dictionary containing image properties analysis
        """
        print(f"\n🔬 ANALYZING IMAGE PROPERTIES ({max_cases} cases)...")
        print("-" * 45)
        
        image_analysis = {
            'dimensions': [],
            'voxel_spacing': [],
            'orientations': [],
            'intensity_stats': {},
            'case_summaries': []
        }
        
        cases_to_analyze = self.case_list[:max_cases]
        
        for case_id in cases_to_analyze:
            case_path = self.ssa_data_path / case_id
            case_summary = {'case_id': case_id}
            
            print(f"🔍 Analyzing: {case_id}")
            
            # Analyze each modality
            for modality in ['t1n', 't1c', 't2w', 't2f']:  # Skip seg for now
                file_path = case_path / f"{case_id}-{modality}.nii.gz"
                
                if file_path.exists():
                    try:
                        # Load NIfTI file
                        nii_img = nib.load(file_path)
                        img_data = nii_img.get_fdata()
                        
                        # Get image properties
                        dimensions = img_data.shape
                        voxel_spacing = nii_img.header.get_zooms()[:3]
                        orientation = nib.aff2axcodes(nii_img.affine)
                        
                        # Store properties
                        image_analysis['dimensions'].append(dimensions)
                        image_analysis['voxel_spacing'].append(voxel_spacing)
                        image_analysis['orientations'].append(orientation)
                        
                        # Intensity statistics (for brain region only)
                        brain_mask = img_data > 0
                        if np.any(brain_mask):
                            brain_intensities = img_data[brain_mask]
                            intensity_stats = {
                                'mean': float(np.mean(brain_intensities)),
                                'std': float(np.std(brain_intensities)),
                                'min': float(np.min(brain_intensities)),
                                'max': float(np.max(brain_intensities)),
                                'percentiles': {
                                    '5': float(np.percentile(brain_intensities, 5)),
                                    '95': float(np.percentile(brain_intensities, 95))
                                }
                            }
                            
                            if modality not in image_analysis['intensity_stats']:
                                image_analysis['intensity_stats'][modality] = []
                            image_analysis['intensity_stats'][modality].append(intensity_stats)
                        
                        case_summary[f'{modality}_shape'] = dimensions
                        case_summary[f'{modality}_spacing'] = voxel_spacing
                        
                    except Exception as e:
                        print(f"   ⚠️ Error loading {modality}: {str(e)}")
                        case_summary[f'{modality}_error'] = str(e)
            
            image_analysis['case_summaries'].append(case_summary)
        
        self.analysis_results['image_properties'] = image_analysis
        
        # Print summary statistics
        if image_analysis['dimensions']:
            dims_array = np.array(image_analysis['dimensions'])
            spacing_array = np.array(image_analysis['voxel_spacing'])
            
            print(f"\n📊 Image Properties Summary:")
            print(f"   📐 Dimensions: {dims_array[0]} (typical)")
            print(f"   📏 Voxel Spacing: {spacing_array[0]} mm (typical)")
            print(f"   🧭 Orientation: {image_analysis['orientations'][0]} (typical)")
        
        return image_analysis
    
    def analyze_segmentation_labels(self, max_cases: int = 5) -> Dict:
        """
        Analyze segmentation labels and tumor characteristics
        
        Args:
            max_cases: Maximum number of cases to analyze
            
        Returns:
            Dictionary containing segmentation analysis results
        """
        print(f"\n🎯 ANALYZING SEGMENTATION LABELS ({max_cases} cases)...")
        print("-" * 50)
        
        seg_analysis = {
            'label_distributions': [],
            'tumor_volumes': [],
            'tumor_statistics': {},
            'label_values': set(),
            'case_details': []
        }
        
        cases_to_analyze = self.case_list[:max_cases]
        
        for case_id in cases_to_analyze:
            case_path = self.ssa_data_path / case_id
            seg_file = case_path / f"{case_id}-seg.nii.gz"
            
            if seg_file.exists():
                try:
                    print(f"🔍 Analyzing segmentation: {case_id}")
                    
                    # Load segmentation
                    seg_nii = nib.load(seg_file)
                    seg_data = seg_nii.get_fdata()
                    voxel_volume = np.prod(seg_nii.header.get_zooms()[:3])  # mm³
                    
                    # Get unique labels
                    unique_labels = np.unique(seg_data)
                    seg_analysis['label_values'].update(unique_labels)
                    
                    # Calculate label distribution
                    label_counts = {}
                    label_volumes = {}
                    
                    for label in unique_labels:
                        count = np.sum(seg_data == label)
                        volume_mm3 = count * voxel_volume
                        volume_ml = volume_mm3 / 1000  # Convert to mL
                        
                        label_counts[int(label)] = int(count)
                        label_volumes[int(label)] = round(volume_ml, 2)
                    
                    seg_analysis['label_distributions'].append(label_counts)
                    seg_analysis['tumor_volumes'].append(label_volumes)
                    
                    case_detail = {
                        'case_id': case_id,
                        'labels_present': [int(l) for l in unique_labels],
                        'label_counts': label_counts,
                        'tumor_volumes_ml': label_volumes,
                        'total_tumor_volume': sum([v for k, v in label_volumes.items() if k > 0])
                    }
                    
                    seg_analysis['case_details'].append(case_detail)
                    
                    print(f"   📊 Labels found: {[int(l) for l in unique_labels]}")
                    print(f"   🧠 Total tumor volume: {case_detail['total_tumor_volume']:.1f} mL")
                    
                except Exception as e:
                    print(f"   ⚠️ Error analyzing {case_id}: {str(e)}")
        
        # Calculate overall statistics
        if seg_analysis['case_details']:
            all_tumor_volumes = [case['total_tumor_volume'] for case in seg_analysis['case_details']]
            seg_analysis['tumor_statistics'] = {
                'mean_volume_ml': round(np.mean(all_tumor_volumes), 2),
                'std_volume_ml': round(np.std(all_tumor_volumes), 2),
                'min_volume_ml': round(min(all_tumor_volumes), 2),
                'max_volume_ml': round(max(all_tumor_volumes), 2),
                'median_volume_ml': round(np.median(all_tumor_volumes), 2)
            }
        
        self.analysis_results['segmentation'] = seg_analysis
        
        print(f"\n✅ Segmentation Analysis Complete:")
        print(f"   🏷️ Unique Labels: {sorted(list(seg_analysis['label_values']))}")
        if seg_analysis['tumor_statistics']:
            stats = seg_analysis['tumor_statistics']
            print(f"   📊 Tumor Volume Range: {stats['min_volume_ml']}-{stats['max_volume_ml']} mL")
            print(f"   📈 Mean Tumor Volume: {stats['mean_volume_ml']} ± {stats['std_volume_ml']} mL")
        
        return seg_analysis
    
    def create_summary_report(self) -> Dict:
        """
        Create a comprehensive summary report of the SSA dataset
        
        Returns:
            Dictionary containing the complete analysis summary
        """
        print("\n📋 CREATING SUMMARY REPORT...")
        print("-" * 30)
        
        summary_report = {
            'analysis_date': datetime.now().isoformat(),
            'dataset_path': str(self.ssa_data_path),
            'total_cases_found': len(self.case_list),
            'analysis_results': self.analysis_results,
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if 'structure' in self.analysis_results:
            structure = self.analysis_results['structure']
            if structure['incomplete_cases'] > 0:
                summary_report['recommendations'].append(
                    f"⚠️ Found {structure['incomplete_cases']} incomplete cases - consider quality filtering"
                )
        
        if 'segmentation' in self.analysis_results:
            seg = self.analysis_results['segmentation']
            if seg['label_values']:
                labels = sorted(list(seg['label_values']))
                summary_report['recommendations'].append(
                    f"🏷️ Label mapping needed for: {labels}"
                )
        
        # Save report to file
        report_path = Path("SSA_Type") / "ssa_dataset_analysis_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        print(f"✅ Summary report saved to: {report_path}")
        
        return summary_report
    
    def run_complete_analysis(self) -> Dict:
        """
        Run the complete SSA dataset analysis pipeline
        
        Returns:
            Complete analysis results
        """
        print("🚀 STARTING COMPLETE SSA DATASET ANALYSIS")
        print("=" * 60)
        
        # Step 1: Discover cases
        self.discover_cases()
        
        if not self.case_list:
            print("❌ No SSA cases found! Please check the dataset path.")
            return {}
        
        # Step 2: Analyze structure
        self.analyze_case_structure()
        
        # Step 3: Analyze image properties
        self.analyze_image_properties(max_cases=5)
        
        # Step 4: Analyze segmentation
        self.analyze_segmentation_labels(max_cases=5)
        
        # Step 5: Create summary report
        summary = self.create_summary_report()
        
        print("\n🎉 SSA DATASET ANALYSIS COMPLETE!")
        print("=" * 45)
        print("📁 Generated Files:")
        print("   - ssa_dataset_analysis_report.json")
        print("\n✅ Ready for next phase: Preprocessing Pipeline Development")
        
        return summary


def main():
    """
    Main function to run SSA dataset analysis
    """
    # Define SSA dataset path
    ssa_dataset_path = "f:/Projects/BrainTumorDetector/archive/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
    
    # Initialize explorer
    explorer = SSADatasetExplorer(ssa_dataset_path)
    
    # Run complete analysis
    results = explorer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    results = main()
