#!/usr/bin/env python3
"""
🧠 SSA Brain Tumor Segmentation - Inference & Visualization Demo
==============================================================

This module demonstrates the trained SSA model's inference capabilities
with comprehensive visualization of:
- Original brain MRI sequences
- Ground truth tumor masks
- Model predictions
- Overlay comparisons
- Performance metrics per slice

"""

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
import os
import json
from datetime import datetime
import math
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from scipy import ndimage

# Import our model
import sys
sys.path.append('SSA_Type')
from ssa_model import SSABrainTumorUNet3D, SSAModelManager

class SSAInferenceDemo:
    """Comprehensive inference demonstration for SSA brain tumor segmentation"""
    
    def __init__(self, model_path, device='cuda'):
        """Initialize the inference demo
        
        Args:
            model_path: Path to trained model
            device: Computing device (cuda/cpu)
        """
        self.device = device
        self.model_path = model_path
        
        # Load the trained model
        print("🔄 Loading trained SSA model...")
        self.model = SSABrainTumorUNet3D(in_channels=4, out_channels=4)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model loaded from checkpoint successfully!")
            else:
                self.model.load_state_dict(checkpoint)
                print("✅ Model loaded successfully!")
        else:
            print(f"❌ Model not found at: {model_path}")
            return
            
        self.model.to(device)
        self.model.eval()
        
        # Create custom colormap for tumor visualization
        self.tumor_colors = ['black', 'red', 'green', 'blue', 'yellow']
        self.tumor_cmap = ListedColormap(self.tumor_colors[:4])

        # Geometry from the last loaded sample (set in load_ssa_sample)
        self.last_spacing = None
        
        print(f"🧠 SSA Inference Demo initialized on {device}")
    
    def load_ssa_sample(self, sample_path):
        """Load SSA sample data for inference
        
        Args:
            sample_path: Path to SSA sample directory
            
        Returns:
            dict: Loaded MRI sequences and segmentation
        """
        print(f"📂 Loading SSA sample from: {sample_path}")
        
        # Expected file patterns for SSA data
        modalities = ['t1n', 't1c', 't2w', 't2f']
        
        data = {
            'spacing': None,
            'affine': None,
        }
        sample_files = list(Path(sample_path).glob("*.nii.gz"))
        
        # Load each modality
        for modality in modalities:
            modality_files = [f for f in sample_files if modality in f.name.lower()]
            if modality_files:
                file_path = modality_files[0]
                img = nib.load(str(file_path))
                data[modality] = img.get_fdata()
                # Store reference geometry from the first available modality
                if data.get('spacing') is None:
                    data['spacing'] = tuple(float(z) for z in img.header.get_zooms()[:3])
                    data['affine'] = img.affine
                print(f"  ✅ {modality.upper()}: {data[modality].shape}")
            else:
                print(f"  ❌ {modality.upper()}: Not found")
                
        # Load segmentation if available
        seg_files = [f for f in sample_files if 'seg' in f.name.lower()]
        if seg_files:
            seg_img = nib.load(str(seg_files[0]))
            data['segmentation'] = seg_img.get_fdata()
            if data.get('spacing') is None:
                data['spacing'] = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
                data['affine'] = seg_img.affine
            print(f"  ✅ Segmentation: {data['segmentation'].shape}")
            
            # Check unique labels
            unique_labels = np.unique(data['segmentation'])
            print(f"  📊 Segmentation labels: {unique_labels}")
        else:
            print(f"  ⚠️ No segmentation found")
            data['segmentation'] = None
            
        self.last_spacing = data.get('spacing')
        return data

    def calculate_tumor_parameters(
        self,
        prediction: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> Dict:
        """Compute clinically useful tumor parameters from a 3D label map.

        Assumes labels:
        0 background
        1 necrotic / non-enhancing tumor core
        2 edema
        3 enhancing tumor
        """
        sx, sy, sz = (spacing if spacing is not None else (1.0, 1.0, 1.0))
        voxel_vol_mm3 = float(sx * sy * sz)
        voxel_area_mm2 = float(sx * sy)

        pred = prediction.astype(np.int32)
        masks = {
            'NCR_NET': pred == 1,
            'ED': pred == 2,
            'ET': pred == 3,
        }

        wt_mask = pred > 0
        tc_mask = (pred == 1) | (pred == 3)
        et_mask = pred == 3

        def _vol_cm3(mask: np.ndarray) -> float:
            return float(mask.sum() * voxel_vol_mm3 / 1000.0)

        def _max_area_slice_cm2(mask: np.ndarray) -> Tuple[int, float]:
            if mask.ndim != 3 or mask.shape[2] == 0:
                return 0, 0.0
            areas = mask.sum(axis=(0, 1)).astype(np.float64) * (voxel_area_mm2 / 100.0)
            idx = int(np.argmax(areas))
            return idx, float(areas[idx])

        def _centroid(mask: np.ndarray) -> Dict:
            if not np.any(mask):
                return {'voxel': None, 'mm': None}
            coords = np.column_stack(np.where(mask))  # (N, 3) in (x,y,z)
            c = coords.mean(axis=0)
            voxel = (float(c[0]), float(c[1]), float(c[2]))
            mm = (float(c[0] * sx), float(c[1] * sy), float(c[2] * sz))
            return {'voxel': voxel, 'mm': mm}

        def _bbox_mm(mask: np.ndarray) -> Dict:
            if not np.any(mask):
                return {'min_voxel': None, 'max_voxel': None, 'size_mm': None}
            xs, ys, zs = np.where(mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            z0, z1 = int(zs.min()), int(zs.max())
            size_mm = (
                float((x1 - x0 + 1) * sx),
                float((y1 - y0 + 1) * sy),
                float((z1 - z0 + 1) * sz),
            )
            return {
                'min_voxel': (x0, y0, z0),
                'max_voxel': (x1, y1, z1),
                'size_mm': size_mm,
            }

        def _components(mask: np.ndarray) -> int:
            if not np.any(mask):
                return 0
            structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
            _, n = ndimage.label(mask, structure=structure)
            return int(n)

        wt_vol = _vol_cm3(wt_mask)
        tc_vol = _vol_cm3(tc_mask)
        et_vol = _vol_cm3(et_mask)
        ed_vol = _vol_cm3(masks['ED'])
        ncr_vol = _vol_cm3(masks['NCR_NET'])

        max_slice_wt, max_area_wt = _max_area_slice_cm2(wt_mask)
        tumor_slices = int(np.count_nonzero(wt_mask.sum(axis=(0, 1)) > 0))

        eq_diam_mm = None
        if wt_vol > 0:
            wt_vol_mm3 = wt_vol * 1000.0
            eq_diam_mm = float(2.0 * ((3.0 * wt_vol_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0))

        enhancing_fraction = float(et_vol / wt_vol) if wt_vol > 0 else 0.0
        edema_to_core = float(ed_vol / tc_vol) if tc_vol > 0 else 0.0
        necrotic_fraction_core = float(ncr_vol / tc_vol) if tc_vol > 0 else 0.0

        params = {
            'spacing_mm': (float(sx), float(sy), float(sz)),
            'voxel_volume_mm3': voxel_vol_mm3,
            'volumes_cm3': {
                'WT': wt_vol,
                'TC': tc_vol,
                'ET': et_vol,
                'ED': ed_vol,
                'NCR_NET': ncr_vol,
            },
            'ratios': {
                'enhancing_fraction_ET_over_WT': enhancing_fraction,
                'edema_to_core_ED_over_TC': edema_to_core,
                'necrotic_fraction_NCR_over_TC': necrotic_fraction_core,
            },
            'max_cross_section_WT': {
                'slice_index': max_slice_wt,
                'area_cm2': max_area_wt,
            },
            'slices_with_tumor': tumor_slices,
            'centroid_WT': _centroid(wt_mask),
            'bbox_WT': _bbox_mm(wt_mask),
            'connected_components_WT': _components(wt_mask),
            'equivalent_spherical_diameter_mm_WT': eq_diam_mm,
        }

        return params
    
    def preprocess_for_inference(self, data):
        """Preprocess data for model inference
        
        Args:
            data: Raw MRI data dictionary
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for inference
        """
        modalities = ['t1n', 't1c', 't2w', 't2f']
        
        # Determine a reference shape for missing modalities
        reference_volume = None
        for modality in modalities:
            if modality in data and data[modality] is not None:
                reference_volume = data[modality]
                break
        if reference_volume is None:
            raise ValueError("No MRI modalities found in sample folder")

        # Stack modalities
        volume_list = []
        for modality in modalities:
            if modality in data:
                volume = data[modality]
                
                # Normalize to [0, 1]
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
                volume_list.append(volume)
            else:
                # Create dummy volume if modality missing
                volume_list.append(np.zeros_like(reference_volume))
        
        # Stack and convert to tensor
        input_volume = np.stack(volume_list, axis=0)  # Shape: (4, H, W, D)
        input_tensor = torch.from_numpy(input_volume).float()
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)  # Shape: (1, 4, H, W, D)
        
        print(f"📊 Preprocessed tensor shape: {input_tensor.shape}")
        return input_tensor
    
    def run_inference(self, input_tensor):
        """Run model inference
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            numpy.ndarray: Predicted segmentation
        """
        print("🔮 Running model inference...")
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            
            # Forward pass
            outputs = self.model(input_tensor)
            
            # Convert to probabilities and get predictions
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Convert to numpy
            predictions_np = predictions.cpu().numpy()[0]  # Remove batch dimension
            probabilities_np = probabilities.cpu().numpy()[0]
            
        print(f"✅ Inference complete. Prediction shape: {predictions_np.shape}")
        print(f"📊 Predicted labels: {np.unique(predictions_np)}")
        
        return predictions_np, probabilities_np
    
    def calculate_metrics(self, prediction, ground_truth):
        """Calculate segmentation metrics
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth segmentation
            
        Returns:
            dict: Calculated metrics
        """
        if ground_truth is None:
            return None
            
        print("📊 Calculating segmentation metrics...")
        
        # Handle SSA label mapping (label 4 -> label 3)
        gt_mapped = ground_truth.copy()
        gt_mapped[gt_mapped == 4] = 3
        
        metrics = {}

        # Local helper: HD95 for binary masks
        def _hd95_binary(a: np.ndarray, b: np.ndarray, spacing_mm=(1.0, 1.0, 1.0)) -> float:
            a = a.astype(bool)
            b = b.astype(bool)

            if not np.any(a) and not np.any(b):
                return 0.0
            if not np.any(a) or not np.any(b):
                return float('nan')

            structure = ndimage.generate_binary_structure(3, 1)
            a_er = ndimage.binary_erosion(a, structure=structure, iterations=1)
            b_er = ndimage.binary_erosion(b, structure=structure, iterations=1)
            a_surf = a ^ a_er
            b_surf = b ^ b_er

            # Distance to the nearest surface voxel
            dt_b = ndimage.distance_transform_edt(~b_surf, sampling=spacing_mm)
            dt_a = ndimage.distance_transform_edt(~a_surf, sampling=spacing_mm)
            dists_ab = dt_b[a_surf]
            dists_ba = dt_a[b_surf]
            if dists_ab.size == 0 or dists_ba.size == 0:
                return float('nan')

            all_dists = np.concatenate([dists_ab, dists_ba]).astype(np.float64)
            return float(np.percentile(all_dists, 95))
        
        # Calculate Dice score + HD95 for each class
        unique_labels = np.unique(gt_mapped)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        dice_scores = []
        hd95_scores = []

        spacing_mm = (1.0, 1.0, 1.0)
        # If caller attached spacing in the instance (set in load_ssa_sample), use it.
        # run_complete_demo calls calculate_metrics; we store spacing on self for access.
        if hasattr(self, 'last_spacing') and self.last_spacing is not None:
            spacing_mm = tuple(float(x) for x in self.last_spacing)

        for label in unique_labels:
            pred_mask = (prediction == label).astype(float)
            gt_mask = (gt_mapped == label).astype(float)
            
            intersection = np.sum(pred_mask * gt_mask)
            union = np.sum(pred_mask) + np.sum(gt_mask)
            
            if union > 0:
                dice = 2.0 * intersection / union
            else:
                dice = 1.0  # Perfect score if both masks are empty
                
            dice_scores.append(dice)
            metrics[f'dice_class_{int(label)}'] = dice

            hd95 = _hd95_binary(prediction == label, gt_mapped == label, spacing_mm=spacing_mm)
            hd95_scores.append(hd95)
            metrics[f'hd95_class_{int(label)}'] = hd95
            
        # Overall metrics
        metrics['mean_dice'] = np.mean(dice_scores)
        metrics['num_classes'] = len(unique_labels)

        # Mean HD95 across present classes (ignoring NaNs)
        if hd95_scores:
            metrics['mean_hd95_mm'] = float(np.nanmean(np.array(hd95_scores, dtype=np.float64)))
        else:
            metrics['mean_hd95_mm'] = float('nan')

        # BraTS-style region aggregates (binary)
        pred_wt = prediction > 0
        gt_wt = gt_mapped > 0
        pred_tc = (prediction == 1) | (prediction == 3)
        gt_tc = (gt_mapped == 1) | (gt_mapped == 3)
        pred_et = prediction == 3
        gt_et = gt_mapped == 3

        metrics['hd95_WT_mm'] = _hd95_binary(pred_wt, gt_wt, spacing_mm=spacing_mm)
        metrics['hd95_TC_mm'] = _hd95_binary(pred_tc, gt_tc, spacing_mm=spacing_mm)
        metrics['hd95_ET_mm'] = _hd95_binary(pred_et, gt_et, spacing_mm=spacing_mm)
        
        print(f"✅ Metrics calculated:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'hd95' in key:
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
            
        return metrics
    
    def create_comprehensive_visualization(self, data, prediction, probabilities, metrics=None, tumor_params=None):
        """Create comprehensive visualization of segmentation results
        
        Args:
            data: Original MRI data
            prediction: Model prediction
            probabilities: Model output probabilities  
            metrics: Calculated metrics
        """
        print("🎨 Creating comprehensive visualization...")
        
        # Get middle slices for visualization
        _, _, depth = prediction.shape
        middle_slice = depth // 2
        slices_to_show = [
            max(0, middle_slice - 10),
            middle_slice,
            min(depth - 1, middle_slice + 10)
        ]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
        
        modalities = ['t1n', 't1c', 't2w', 't2f']
        
        for slice_idx, slice_num in enumerate(slices_to_show):
            
            # Original modalities
            for mod_idx, modality in enumerate(modalities):
                ax = fig.add_subplot(gs[slice_idx, mod_idx])
                
                if modality in data:
                    img_slice = data[modality][:, :, slice_num]
                    ax.imshow(img_slice, cmap='gray')
                    ax.set_title(f'{modality.upper()} - Slice {slice_num}', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{modality.upper()} - N/A', fontsize=10)
                
                ax.axis('off')
            
            # Ground truth segmentation
            ax_gt = fig.add_subplot(gs[slice_idx, 4])
            if data['segmentation'] is not None:
                gt_slice = data['segmentation'][:, :, slice_num]
                # Map label 4 to 3 for visualization
                gt_slice_mapped = gt_slice.copy()
                gt_slice_mapped[gt_slice_mapped == 4] = 3
                
                ax_gt.imshow(gt_slice_mapped, cmap=self.tumor_cmap, vmin=0, vmax=3)
                ax_gt.set_title(f'Ground Truth - Slice {slice_num}', fontsize=10)
            else:
                ax_gt.text(0.5, 0.5, 'No GT', ha='center', va='center', transform=ax_gt.transAxes)
                ax_gt.set_title(f'Ground Truth - N/A', fontsize=10)
            ax_gt.axis('off')
            
            # Model prediction
            ax_pred = fig.add_subplot(gs[slice_idx, 5])
            pred_slice = prediction[:, :, slice_num]
            ax_pred.imshow(pred_slice, cmap=self.tumor_cmap, vmin=0, vmax=3)
            ax_pred.set_title(f'Prediction - Slice {slice_num}', fontsize=10)
            ax_pred.axis('off')
        
        # Metrics summary
        ax_metrics = fig.add_subplot(gs[3, :3])
        ax_metrics.axis('off')
        
        if metrics:
            metrics_text = f"""
🏆 SEGMENTATION PERFORMANCE METRICS

Overall Performance:
• Mean Dice Score: {metrics['mean_dice']:.4f} ({metrics['mean_dice']*100:.2f}%)
• Mean HD95: {metrics.get('mean_hd95_mm', float('nan')):.2f} mm
• Number of Classes: {metrics['num_classes']}

Class-wise Dice Scores:"""
            
            for key, value in metrics.items():
                if key.startswith('dice_class_'):
                    class_num = key.split('_')[-1]
                    metrics_text += f"\n• Class {class_num}: {value:.4f} ({value*100:.2f}%)"

            # HD95 per class (if available)
            if any(k.startswith('hd95_class_') for k in metrics.keys()):
                metrics_text += "\n\nClass-wise HD95 (mm):"
                for key, value in metrics.items():
                    if key.startswith('hd95_class_'):
                        class_num = key.split('_')[-1]
                        if value == value:  # not NaN
                            metrics_text += f"\n• Class {class_num}: {value:.2f}"
                        else:
                            metrics_text += f"\n• Class {class_num}: N/A"

            # Aggregates (WT/TC/ET)
            if any(k in metrics for k in ('hd95_WT_mm', 'hd95_TC_mm', 'hd95_ET_mm')):
                metrics_text += "\n\nRegion HD95 (mm):"
                metrics_text += f"\n• WT: {metrics.get('hd95_WT_mm', float('nan')):.2f}"
                metrics_text += f"\n• TC: {metrics.get('hd95_TC_mm', float('nan')):.2f}"
                metrics_text += f"\n• ET: {metrics.get('hd95_ET_mm', float('nan')):.2f}"
            
            metrics_text += f"""

Clinical Assessment:
• Status: {'✅ EXCELLENT' if metrics['mean_dice'] >= 0.8 else '✅ GOOD' if metrics['mean_dice'] >= 0.7 else '⚠️ FAIR'}
• Clinical Grade: {'Research-grade' if metrics['mean_dice'] >= 0.8 else 'Clinical-grade' if metrics['mean_dice'] >= 0.7 else 'Acceptable'}
"""
        else:
            metrics_text = """
🏆 INFERENCE DEMONSTRATION

No ground truth available for quantitative evaluation.
Showing qualitative segmentation results.

Visual Assessment:
• Check tumor boundary definition
• Evaluate anatomical consistency  
• Assess false positive/negative regions
"""
        
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=11, verticalalignment='top', 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Color legend
        ax_legend = fig.add_subplot(gs[3, 3:])
        ax_legend.axis('off')
        
        legend_text = """
🎨 SEGMENTATION COLOR LEGEND

Tumor Labels:
🖤 Label 0: Background (Black)
🔴 Label 1: Necrotic/Non-enhancing Tumor (Red)  
🟢 Label 2: Peritumoral/Edema (Green)
🔵 Label 3: Enhancing Tumor (Blue)

Note: SSA label 4 is mapped to label 3 for model compatibility
"""

        if tumor_params:
            v = tumor_params.get('volumes_cm3', {})
            r = tumor_params.get('ratios', {})
            bbox = (tumor_params.get('bbox_WT') or {}).get('size_mm')
            centroid_mm = (tumor_params.get('centroid_WT') or {}).get('mm')
            max_cs = tumor_params.get('max_cross_section_WT', {})

            bbox_text = f"{bbox[0]:.1f}×{bbox[1]:.1f}×{bbox[2]:.1f} mm" if bbox else "N/A"
            centroid_text = f"({centroid_mm[0]:.1f}, {centroid_mm[1]:.1f}, {centroid_mm[2]:.1f}) mm" if centroid_mm else "N/A"
            eqd = tumor_params.get('equivalent_spherical_diameter_mm_WT')
            eqd_text = f"{eqd:.1f} mm" if eqd is not None else "N/A"

            legend_text += f"""

📦 TUMOR PARAMETERS (Prediction)

Volumes (cm³):
• WT (Whole): {v.get('WT', 0.0):.2f}
• TC (Core):  {v.get('TC', 0.0):.2f}
• ET (Enh.):  {v.get('ET', 0.0):.2f}
• ED (Edema): {v.get('ED', 0.0):.2f}

Ratios:
• ET/WT: {r.get('enhancing_fraction_ET_over_WT', 0.0):.2f}
• ED/TC: {r.get('edema_to_core_ED_over_TC', 0.0):.2f}

Extent:
• Max WT area: {max_cs.get('area_cm2', 0.0):.2f} cm² (slice {max_cs.get('slice_index', 0)})
• Tumor slices: {tumor_params.get('slices_with_tumor', 0)}
• WT bbox: {bbox_text}
• WT centroid: {centroid_text}
• WT components: {tumor_params.get('connected_components_WT', 0)}
• Eq. sphere diam (WT): {eqd_text}
"""
        
        ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Main title
        fig.suptitle('SSA Brain Tumor Segmentation - Inference Results Demonstration', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save visualization
        output_dir = Path('SSA_Type/visualizations/inference')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'ssa_inference_demonstration.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Visualization saved: {output_path}")
        return output_path
    
    def create_3d_volume_analysis(self, prediction, ground_truth=None, spacing=None):
        """Create 3D volume analysis and statistics
        
        Args:
            prediction: Model prediction volume
            ground_truth: Ground truth volume (optional)
        """
        print("📊 Creating 3D volume analysis...")

        sx, sy, sz = (spacing if spacing is not None else (1.0, 1.0, 1.0))
        voxel_vol_mm3 = float(sx * sy * sz)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Volume statistics
        unique_pred, counts_pred = np.unique(prediction, return_counts=True)
        
        # Class distribution
        ax = axes[0, 0]
        colors = ['black', 'red', 'green', 'blue'][:len(unique_pred)]
        bars = ax.bar([f'Class {int(i)}' for i in unique_pred], counts_pred, color=colors, alpha=0.7)
        ax.set_title('Predicted Class Distribution', fontweight='bold')
        ax.set_ylabel('Voxel Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts_pred):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_pred)*0.01,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Slice-wise tumor volume
        ax = axes[0, 1]
        slice_volumes = []
        for z in range(prediction.shape[2]):
            tumor_voxels = np.sum(prediction[:, :, z] > 0)
            slice_volumes.append(tumor_voxels)
        
        ax.plot(slice_volumes, 'b-', linewidth=2)
        ax.set_title('Tumor Volume per Slice', fontweight='bold')
        ax.set_xlabel('Slice Number')
        ax.set_ylabel('Tumor Voxels')
        ax.grid(True, alpha=0.3)
        
        # 3D tumor center analysis
        ax = axes[0, 2]
        tumor_mask = prediction > 0
        if np.any(tumor_mask):
            # Find center of mass
            coords = np.where(tumor_mask)
            center_x = np.mean(coords[0])
            center_y = np.mean(coords[1]) 
            center_z = np.mean(coords[2])
            
            ax.scatter(center_y, center_x, s=100, c='red', marker='x', linewidth=3)
            ax.set_title(f'Tumor Center of Mass\n({center_x:.1f}, {center_y:.1f}, {center_z:.1f})', fontweight='bold')
            ax.set_xlabel('Y Coordinate')
            ax.set_ylabel('X Coordinate')
            ax.grid(True, alpha=0.3)
        
        # If ground truth available, show comparison
        if ground_truth is not None:
            # Map GT labels
            gt_mapped = ground_truth.copy()
            gt_mapped[gt_mapped == 4] = 3
            
            # GT class distribution
            ax = axes[1, 0]
            unique_gt, counts_gt = np.unique(gt_mapped, return_counts=True)
            bars = ax.bar([f'Class {int(i)}' for i in unique_gt], counts_gt, color=colors[:len(unique_gt)], alpha=0.7)
            ax.set_title('Ground Truth Class Distribution', fontweight='bold')
            ax.set_ylabel('Voxel Count')
            
            for bar, count in zip(bars, counts_gt):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_gt)*0.01,
                       f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            # Volume comparison
            ax = axes[1, 1]
            pred_volume = np.sum(prediction > 0)
            gt_volume = np.sum(gt_mapped > 0)

            pred_cm3 = (pred_volume * voxel_vol_mm3) / 1000.0
            gt_cm3 = (gt_volume * voxel_vol_mm3) / 1000.0
            volumes = [pred_cm3, gt_cm3]
            labels = ['Prediction', 'Ground Truth']
            bars = ax.bar(labels, volumes, color=['blue', 'orange'], alpha=0.7)
            ax.set_title('Total Tumor Volume Comparison', fontweight='bold')
            ax.set_ylabel('Volume (cm³)')
            
            for bar, volume in zip(bars, volumes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volumes)*0.01,
                       f'{volume:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Overlap analysis
            ax = axes[1, 2]
            intersection = np.sum((prediction > 0) & (gt_mapped > 0))
            union = np.sum((prediction > 0) | (gt_mapped > 0))
            
            overlap_metrics = {
                'Intersection': intersection,
                'Union': union,
                'IoU': intersection / union if union > 0 else 0
            }
            
            ax.bar(overlap_metrics.keys(), overlap_metrics.values(), 
                  color=['green', 'orange', 'purple'], alpha=0.7)
            ax.set_title('Overlap Analysis', fontweight='bold')
            ax.set_ylabel('Voxel Count / Score')
            
            for i, (key, value) in enumerate(overlap_metrics.items()):
                if key == 'IoU':
                    label_text = f'{value:.3f}'
                else:
                    label_text = f'{int(value):,}'
                ax.text(i, value + max(overlap_metrics.values())*0.01,
                       label_text, ha='center', va='bottom', fontweight='bold')
        else:
            # Fill remaining subplots with info
            for i in range(3):
                ax = axes[1, i]
                ax.axis('off')
                ax.text(0.5, 0.5, 'Ground Truth\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, bbox=dict(boxstyle="round", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save analysis
        output_dir = Path('SSA_Type/visualizations/inference')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'ssa_3d_volume_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 3D Analysis saved: {output_path}")
        return output_path
    
    def run_complete_demo(self, sample_path):
        """Run complete inference demonstration
        
        Args:
            sample_path: Path to SSA sample data
        """
        print("🚀 Starting Complete SSA Inference Demonstration")
        print("=" * 60)
        
        # Load sample data
        data = self.load_ssa_sample(sample_path)
        if not data:
            print("❌ Failed to load sample data")
            return
        
        # Preprocess for inference
        input_tensor = self.preprocess_for_inference(data)
        
        # Run inference
        prediction, probabilities = self.run_inference(input_tensor)

        # Compute tumor parameters (from prediction)
        tumor_params = self.calculate_tumor_parameters(prediction, data.get('spacing'))
        
        # Calculate metrics if ground truth available
        metrics = None
        if data['segmentation'] is not None:
            metrics = self.calculate_metrics(prediction, data['segmentation'])
        
        # Create visualizations
        viz_path = self.create_comprehensive_visualization(data, prediction, probabilities, metrics, tumor_params=tumor_params)
        analysis_path = self.create_3d_volume_analysis(prediction, data['segmentation'], spacing=data.get('spacing'))
        
        # Save inference results
        results = {
            'timestamp': datetime.now().isoformat(),
            'sample_path': str(sample_path),
            'prediction_shape': prediction.shape,
            'unique_predictions': np.unique(prediction).tolist(),
            'metrics': metrics,
            'tumor_parameters': tumor_params,
            'spacing': data.get('spacing'),
            'model_path': self.model_path
        }
        
        results_dir = Path('SSA_Type/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = str(results_dir / 'ssa_inference_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n🎊 INFERENCE DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print(f"📊 Visualization: {viz_path}")
        print(f"📈 3D Analysis: {analysis_path}")
        print(f"💾 Results: {results_path}")
        
        if metrics:
            print(f"🏆 Mean Dice Score: {metrics['mean_dice']:.4f} ({metrics['mean_dice']*100:.2f}%)")
        
        return results

def main():
    """Main function to run SSA inference demonstration"""
    
    # Configuration
    model_path = "SSA_Type/models/best_ssa_model.pth"
    sample_path = "archive/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2/BraTS-SSA-00002-000"
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    if device == 'cpu':
        print("⚠️ Running on CPU - inference may be slower but results will be identical")
    
    # Initialize demo
    demo = SSAInferenceDemo(model_path, device)
    
    # Run complete demonstration
    results = demo.run_complete_demo(sample_path)
    
    return results

if __name__ == "__main__":
    main()
