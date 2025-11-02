#!/usr/bin/env python3
"""
🧠 Brain Tumor Classification System using Segmentation Models
==============================================================

This module implements a heuristic-based classification system that leverages
existing segmentation models (Glioma and SSA) to determine tumor type first,
then perform appropriate segmentation.

Concept: A segmentation model trained on one tumor type will produce poor/low-confidence
segmentation when given an image of a different type. We use this behavior to classify.

"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import time
from dataclasses import dataclass
from enum import Enum

# MONAI imports for medical imaging
try:
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Compose, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
        NormalizeIntensityd, ResizeWithPadOrCropd
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("⚠️ MONAI not available - limited functionality")

class TumorType(Enum):
    """Enumeration for tumor types"""
    GLIOMA = "glioma"
    SSA = "ssa"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult:
    """Data class to hold classification results"""
    predicted_type: TumorType
    confidence_score: float
    glioma_score: float
    ssa_score: float
    segmentation_mask: Optional[np.ndarray] = None
    processing_time: float = 0.0
    model_outputs: Optional[Dict] = None

class SegmentationBasedClassifier:
    """
    Brain tumor classifier using existing segmentation models.
    
    This classifier works by running both segmentation models on the input
    and determining which produces a more confident/better segmentation.
    """
    
    def __init__(self, 
                 glioma_model_path: str,
                 ssa_model_path: str,
                 device: str = 'auto'):
        """
        Initialize the classifier with pre-trained segmentation models.
        
        Args:
            glioma_model_path: Path to the trained glioma segmentation model
            ssa_model_path: Path to the trained SSA segmentation model
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.device = self._setup_device(device)
        self.logger = self._setup_logging()
        
        # Initialize models
        self.glioma_model = None
        self.ssa_model = None
        
        # Load models
        self.glioma_model_path = Path(glioma_model_path)
        self.ssa_model_path = Path(ssa_model_path)
        
        self._load_models()
        
        # Classification thresholds (can be tuned)
        self.confidence_threshold = 0.1  # Minimum confidence for classification
        self.ratio_threshold = 1.5       # Minimum ratio between scores for confident classification
        
        self.logger.info(f"✅ Classifier initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        torch_device = torch.device(device)
        
        if torch_device.type == 'cuda' and torch.cuda.is_available():
            print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("💻 Using CPU")
            
        return torch_device
    
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
    
    def _create_unet_model(self, num_classes: int) -> nn.Module:
        """Create a 3D U-Net model matching the training architecture"""
        if MONAI_AVAILABLE:
            # Create a wrapper class that matches the original BraTSUNet structure
            class BraTSUNet(nn.Module):
                def __init__(self, 
                             spatial_dims=3,
                             in_channels=4,
                             out_channels=4,
                             channels=(32, 64, 128, 256, 512),
                             strides=(2, 2, 2, 2),
                             num_res_units=2,
                             norm=Norm.BATCH,
                             dropout=0.1):
                    super().__init__()
                    
                    self.unet = UNet(
                        spatial_dims=spatial_dims,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        channels=channels,
                        strides=strides,
                        num_res_units=num_res_units,
                        norm=norm,
                        dropout=dropout,
                        act='relu'
                    )
                
                def forward(self, x):
                    return self.unet(x)
            
            model = BraTSUNet(
                spatial_dims=3,
                in_channels=4,  # T1, T1ce, T2, FLAIR
                out_channels=num_classes,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
                dropout=0.1
            )
        else:
            # Fallback to basic implementation if MONAI not available
            model = self._create_basic_unet(num_classes)
            
        return model
    
    def _create_basic_unet(self, num_classes: int) -> nn.Module:
        """Create basic U-Net if MONAI is not available"""
        # This would be a simplified version - for now, raise error
        raise ImportError("MONAI required for full functionality. Please install MONAI.")
    
    def _load_models(self):
        """Load pre-trained segmentation models"""
        try:
            # Load Glioma model
            self.logger.info(f"Loading Glioma model from {self.glioma_model_path}")
            # Check file existence before attempting to load to provide a clear error
            if not self.glioma_model_path.exists():
                raise FileNotFoundError(f"Glioma model file not found: {self.glioma_model_path}")
            self.glioma_model = self._create_unet_model(num_classes=4)  # Background + 3 tumor classes
            
            glioma_checkpoint = torch.load(self.glioma_model_path, map_location=self.device)
            
            # Extract state dict and handle potential key mismatches
            if 'model_state_dict' in glioma_checkpoint:
                state_dict = glioma_checkpoint['model_state_dict']
            else:
                state_dict = glioma_checkpoint
            
            # Handle key prefix mismatch (remove "unet." prefix if present)
            corrected_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('unet.'):
                    new_key = key[5:]  # Remove "unet." prefix
                else:
                    new_key = key
                corrected_state_dict[new_key] = value
            
            self.glioma_model.load_state_dict(corrected_state_dict, strict=False)
            self.glioma_model.to(self.device)
            self.glioma_model.eval()
            
            # Load SSA model
            self.logger.info(f"Loading SSA model from {self.ssa_model_path}")
            if not self.ssa_model_path.exists():
                raise FileNotFoundError(f"SSA model file not found: {self.ssa_model_path}")
            self.ssa_model = self._create_unet_model(num_classes=4)  # Assuming same structure
            
            ssa_checkpoint = torch.load(self.ssa_model_path, map_location=self.device)
            
            # Extract state dict and handle potential key mismatches
            if 'model_state_dict' in ssa_checkpoint:
                state_dict = ssa_checkpoint['model_state_dict']
            else:
                state_dict = ssa_checkpoint
            
            # Handle key prefix mismatch (remove "unet." prefix if present)
            corrected_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('unet.'):
                    new_key = key[5:]  # Remove "unet." prefix
                else:
                    new_key = key
                corrected_state_dict[new_key] = value
            
            self.ssa_model.load_state_dict(corrected_state_dict, strict=False)
            self.ssa_model.to(self.device)
            self.ssa_model.eval()
            
            self.logger.info("✅ Both models loaded successfully")
            
        except Exception as e:
            # Log a clearer message and re-raise for the caller to handle
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _preprocess_input(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess input image for model inference.
        
        Args:
            image: Input image array with shape (4, H, W, D) or (H, W, D, 4)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Ensure correct shape (4, H, W, D)
        if image.shape[-1] == 4:
            image = np.transpose(image, (3, 0, 1, 2))
        
        # Normalize intensity values
        image = image.astype(np.float32)
        
        # Simple normalization (can be enhanced based on training preprocessing)
        for i in range(4):
            modality = image[i]
            if modality.max() > modality.min():
                # Z-score normalization
                mean = modality.mean()
                std = modality.std()
                image[i] = (modality - mean) / (std + 1e-8)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        return tensor
    
    def _calculate_confidence_metrics(self, 
                                    segmentation_output: torch.Tensor, 
                                    model_name: str) -> Dict[str, float]:
        """
        Calculate various confidence metrics from segmentation output.
        
        Args:
            segmentation_output: Raw model output (logits or probabilities)
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing confidence metrics
        """
        # Convert to probabilities if needed
        if segmentation_output.max() > 1.0:
            probabilities = torch.softmax(segmentation_output, dim=1)
        else:
            probabilities = segmentation_output
        
        # Remove batch dimension
        probs = probabilities.squeeze(0).cpu().numpy()
        
        # Calculate various confidence metrics
        metrics = {}
        
        # 1. Mean probability of non-background classes
        non_bg_probs = probs[1:]  # Exclude background (class 0)
        metrics['mean_tumor_prob'] = np.mean(non_bg_probs)
        
        # 2. Max probability across all tumor classes
        metrics['max_tumor_prob'] = np.max(non_bg_probs)
        
        # 3. Sum of all tumor probabilities
        metrics['sum_tumor_prob'] = np.sum(non_bg_probs)
        
        # 4. Entropy (uncertainty measure - lower is more confident)
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=0)
        metrics['mean_entropy'] = np.mean(entropy)
        metrics['min_entropy'] = np.min(entropy)
        
        # 5. Predicted tumor volume (number of pixels with tumor prediction)
        pred_mask = np.argmax(probs, axis=0)
        tumor_volume = np.sum(pred_mask > 0)  # Non-background pixels
        total_volume = pred_mask.size
        metrics['tumor_volume_ratio'] = tumor_volume / total_volume
        
        # 6. Confidence score (combination of metrics)
        # Higher mean probability + lower entropy + reasonable volume
        confidence_score = (
            metrics['mean_tumor_prob'] * 0.4 +
            (1 - metrics['mean_entropy'] / 10) * 0.3 +  # Normalize entropy
            metrics['tumor_volume_ratio'] * 0.3
        )
        metrics['confidence_score'] = max(0.0, min(1.0, confidence_score))
        
        return metrics
    
    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        Classify tumor type using segmentation models.
        
        Args:
            image: Input MRI image with 4 modalities, shape (4, H, W, D) or (H, W, D, 4)
            
        Returns:
            ClassificationResult containing prediction and confidence scores
        """
        start_time = time.time()
        
        try:
            # Preprocess input
            input_tensor = self._preprocess_input(image)
            
            # Run both models
            with torch.no_grad():
                # Glioma model inference
                glioma_output = self.glioma_model(input_tensor)
                glioma_metrics = self._calculate_confidence_metrics(glioma_output, "Glioma")
                
                # SSA model inference  
                ssa_output = self.ssa_model(input_tensor)
                ssa_metrics = self._calculate_confidence_metrics(ssa_output, "SSA")
            
            # Extract confidence scores
            glioma_score = glioma_metrics['confidence_score']
            ssa_score = ssa_metrics['confidence_score']
            
            # Make classification decision
            if glioma_score > ssa_score:
                if glioma_score > self.confidence_threshold and \
                   glioma_score / (ssa_score + 1e-8) > self.ratio_threshold:
                    predicted_type = TumorType.GLIOMA
                    best_output = glioma_output
                else:
                    predicted_type = TumorType.UNKNOWN
                    best_output = glioma_output
            else:
                if ssa_score > self.confidence_threshold and \
                   ssa_score / (glioma_score + 1e-8) > self.ratio_threshold:
                    predicted_type = TumorType.SSA
                    best_output = ssa_output
                else:
                    predicted_type = TumorType.UNKNOWN
                    best_output = ssa_output
            
            # Generate segmentation mask from best model
            segmentation_mask = None
            if best_output is not None:
                with torch.no_grad():
                    probs = torch.softmax(best_output, dim=1)
                    pred_mask = torch.argmax(probs, dim=1)
                    segmentation_mask = pred_mask.squeeze(0).cpu().numpy()
            
            # Calculate overall confidence
            confidence_score = max(glioma_score, ssa_score)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ClassificationResult(
                predicted_type=predicted_type,
                confidence_score=confidence_score,
                glioma_score=glioma_score,
                ssa_score=ssa_score,
                segmentation_mask=segmentation_mask,
                processing_time=processing_time,
                model_outputs={
                    'glioma_metrics': glioma_metrics,
                    'ssa_metrics': ssa_metrics
                }
            )
            
            self.logger.info(f"Classification complete: {predicted_type.value} "
                           f"(confidence: {confidence_score:.3f}, time: {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Classification failed: {str(e)}")
            raise
    
    def batch_classify(self, images: List[np.ndarray]) -> List[ClassificationResult]:
        """
        Classify multiple images in batch.
        
        Args:
            images: List of input images
            
        Returns:
            List of classification results
        """
        results = []
        for i, image in enumerate(images):
            self.logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.classify(image)
            results.append(result)
        
        return results
    
    def update_thresholds(self, 
                         confidence_threshold: Optional[float] = None,
                         ratio_threshold: Optional[float] = None):
        """
        Update classification thresholds for fine-tuning.
        
        Args:
            confidence_threshold: Minimum confidence for classification
            ratio_threshold: Minimum ratio between scores for confident classification
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            self.logger.info(f"Updated confidence threshold to {confidence_threshold}")
        
        if ratio_threshold is not None:
            self.ratio_threshold = ratio_threshold  
            self.logger.info(f"Updated ratio threshold to {ratio_threshold}")
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'glioma_model_path': str(self.glioma_model_path),
            'ssa_model_path': str(self.ssa_model_path),
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'ratio_threshold': self.ratio_threshold,
            'monai_available': MONAI_AVAILABLE
        }
        
        if self.glioma_model is not None:
            info['glioma_model_params'] = sum(p.numel() for p in self.glioma_model.parameters())
            
        if self.ssa_model is not None:
            info['ssa_model_params'] = sum(p.numel() for p in self.ssa_model.parameters())
        
        return info