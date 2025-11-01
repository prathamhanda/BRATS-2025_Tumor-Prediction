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
import warnings
warnings.filterwarnings('ignore')

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
        
        data = {}
        sample_files = list(Path(sample_path).glob("*.nii.gz"))
        
        # Load each modality
        for modality in modalities:
            modality_files = [f for f in sample_files if modality in f.name.lower()]
            if modality_files:
                file_path = modality_files[0]
                img = nib.load(str(file_path))
                data[modality] = img.get_fdata()
                print(f"  ✅ {modality.upper()}: {data[modality].shape}")
            else:
                print(f"  ❌ {modality.upper()}: Not found")
                
        # Load segmentation if available
        seg_files = [f for f in sample_files if 'seg' in f.name.lower()]
        if seg_files:
            seg_img = nib.load(str(seg_files[0]))
            data['segmentation'] = seg_img.get_fdata()
            print(f"  ✅ Segmentation: {data['segmentation'].shape}")
            
            # Check unique labels
            unique_labels = np.unique(data['segmentation'])
            print(f"  📊 Segmentation labels: {unique_labels}")
        else:
            print(f"  ⚠️ No segmentation found")
            data['segmentation'] = None
            
        return data
    
    def preprocess_for_inference(self, data):
        """Preprocess data for model inference
        
        Args:
            data: Raw MRI data dictionary
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for inference
        """
        modalities = ['t1n', 't1c', 't2w', 't2f']
        
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
                volume_list.append(np.zeros_like(volume_list[0]))
        
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
        
        # Calculate Dice score for each class
        unique_labels = np.unique(gt_mapped)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        dice_scores = []
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
            
        # Overall metrics
        metrics['mean_dice'] = np.mean(dice_scores)
        metrics['num_classes'] = len(unique_labels)
        
        print(f"✅ Metrics calculated:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
            
        return metrics
    
    def create_comprehensive_visualization(self, data, prediction, probabilities, metrics=None):
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
• Number of Classes: {metrics['num_classes']}

Class-wise Dice Scores:"""
            
            for key, value in metrics.items():
                if key.startswith('dice_class_'):
                    class_num = key.split('_')[-1]
                    metrics_text += f"\n• Class {class_num}: {value:.4f} ({value*100:.2f}%)"
            
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
        
        ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Main title
        fig.suptitle('SSA Brain Tumor Segmentation - Inference Results Demonstration', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save visualization
        output_path = 'SSA_Type/SSA_Type/ssa_inference_demonstration.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Visualization saved: {output_path}")
        return output_path
    
    def create_3d_volume_analysis(self, prediction, ground_truth=None):
        """Create 3D volume analysis and statistics
        
        Args:
            prediction: Model prediction volume
            ground_truth: Ground truth volume (optional)
        """
        print("📊 Creating 3D volume analysis...")
        
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
            
            volumes = [pred_volume, gt_volume]
            labels = ['Prediction', 'Ground Truth']
            bars = ax.bar(labels, volumes, color=['blue', 'orange'], alpha=0.7)
            ax.set_title('Total Tumor Volume Comparison', fontweight='bold')
            ax.set_ylabel('Voxel Count')
            
            for bar, volume in zip(bars, volumes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volumes)*0.01,
                       f'{volume:,}', ha='center', va='bottom', fontweight='bold')
            
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
        output_path = 'SSA_Type/SSA_Type/ssa_3d_volume_analysis.png'
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
        
        # Calculate metrics if ground truth available
        metrics = None
        if data['segmentation'] is not None:
            metrics = self.calculate_metrics(prediction, data['segmentation'])
        
        # Create visualizations
        viz_path = self.create_comprehensive_visualization(data, prediction, probabilities, metrics)
        analysis_path = self.create_3d_volume_analysis(prediction, data['segmentation'])
        
        # Save inference results
        results = {
            'timestamp': datetime.now().isoformat(),
            'sample_path': str(sample_path),
            'prediction_shape': prediction.shape,
            'unique_predictions': np.unique(prediction).tolist(),
            'metrics': metrics,
            'model_path': self.model_path
        }
        
        results_path = 'SSA_Type/SSA_Type/ssa_inference_results.json'
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
    model_path = "SSA_Type/SSA_Type/training_results/best_ssa_model.pth"
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
