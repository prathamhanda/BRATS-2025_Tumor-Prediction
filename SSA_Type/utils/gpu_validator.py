#!/usr/bin/env python3
"""
🔥 GPU Validation Script for SSA Brain Tumor Segmentation
=========================================================

This script validates your NVIDIA GeForce GTX 1650 setup and provides
optimization recommendations for intensive deep learning tasks.

Date: September 7, 2025
"""

import torch
import gc
import psutil
import platform
import subprocess
import sys

def get_gpu_info():
    """Get detailed GPU information"""
    print("🔥 GPU VALIDATION REPORT")
    print("=" * 60)
    
    # Basic CUDA info
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ PyTorch Version: {torch.__version__}")
        print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
        
        # GPU Memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        
        # Compute capability
        compute_cap = torch.cuda.get_device_properties(0)
        print(f"✅ Compute Capability: {compute_cap.major}.{compute_cap.minor}")
        
        # Current memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"📊 Memory Allocated: {allocated:.2f} GB")
            print(f"📊 Memory Reserved: {reserved:.2f} GB")
            print(f"📊 Memory Available: {gpu_memory - reserved:.2f} GB")
            
        return True, gpu_memory, compute_cap
    else:
        print("❌ CUDA not available - using CPU only")
        return False, 0, None

def validate_environment():
    """Validate the complete environment"""
    print("\n🖥️ SYSTEM ENVIRONMENT")
    print("=" * 60)
    
    # System info
    print(f"💻 Platform: {platform.platform()}")
    print(f"🐍 Python Version: {sys.version}")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"🧠 System RAM: {memory.total / (1024**3):.1f} GB")
    print(f"🧠 Available RAM: {memory.available / (1024**3):.1f} GB")
    
    # Check required packages
    print(f"\n📦 REQUIRED PACKAGES")
    print("=" * 60)
    
    required_packages = {
        'torch': torch.__version__,
        'numpy': None,
        'nibabel': None,
        'scipy': None,
        'scikit-image': None,
        'matplotlib': None,
        'tqdm': None
    }
    
    for package, version in required_packages.items():
        try:
            if package == 'torch':
                print(f"✅ {package}: {version}")
            else:
                __import__(package)
                if package == 'numpy':
                    import numpy as np
                    print(f"✅ {package}: {np.__version__}")
                elif package == 'nibabel':
                    import nibabel as nib
                    print(f"✅ {package}: {nib.__version__}")
                elif package == 'scipy':
                    import scipy
                    print(f"✅ {package}: {scipy.__version__}")
                elif package == 'scikit-image':
                    import skimage
                    print(f"✅ {package}: {skimage.__version__}")
                elif package == 'matplotlib':
                    import matplotlib
                    print(f"✅ {package}: {matplotlib.__version__}")
                elif package == 'tqdm':
                    import tqdm
                    print(f"✅ {package}: {tqdm.__version__}")
        except ImportError:
            print(f"❌ {package}: Not installed")

def test_gpu_operations():
    """Test basic GPU operations"""
    if not torch.cuda.is_available():
        print("\n❌ GPU not available - skipping GPU tests")
        return False
        
    print(f"\n🧪 GPU PERFORMANCE TESTS")
    print("=" * 60)
    
    try:
        # Test tensor operations
        device = torch.device('cuda:0')
        
        # Small tensor test
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("✅ Basic tensor operations: PASSED")
        
        # Memory allocation test
        large_tensor = torch.randn(2000, 2000, device=device)
        print("✅ Large tensor allocation: PASSED")
        
        # 3D convolution test (important for brain tumor segmentation)
        conv3d = torch.nn.Conv3d(4, 32, kernel_size=3, padding=1).to(device)
        test_input = torch.randn(1, 4, 64, 64, 64, device=device)
        output = conv3d(test_input)
        print("✅ 3D Convolution operations: PASSED")
        
        # Memory cleanup
        del x, y, z, large_tensor, conv3d, test_input, output
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def recommend_optimizations(gpu_memory):
    """Provide optimization recommendations for GTX 1650"""
    print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS FOR GTX 1650")
    print("=" * 60)
    
    if gpu_memory > 0:
        print(f"🎯 Your GTX 1650 has {gpu_memory:.1f} GB VRAM")
        
        if gpu_memory >= 4.0:
            print("✅ Good! You have sufficient VRAM for brain tumor segmentation")
            print("📊 Recommended settings:")
            print("   - Batch Size: 1-2 (for 128³ patches)")
            print("   - Mixed Precision: Enabled (save ~30% memory)")
            print("   - Gradient Checkpointing: Enabled")
            print("   - Pin Memory: True")
        else:
            print("⚠️ Limited VRAM - use aggressive memory optimization")
            print("📊 Recommended settings:")
            print("   - Batch Size: 1 only")
            print("   - Mixed Precision: REQUIRED")
            print("   - Gradient Checkpointing: REQUIRED")
            print("   - Smaller patch size: Consider 96³ instead of 128³")
            
        print("\n🔧 Performance Optimizations:")
        print("   - Use DataLoader with num_workers=2-4")
        print("   - Enable torch.backends.cudnn.benchmark=True")
        print("   - Use torch.compile() for PyTorch 2.0+")
        print("   - Implement gradient accumulation for larger effective batch size")
        
    print("\n💡 Memory Management Tips:")
    print("   - Clear cache regularly: torch.cuda.empty_cache()")
    print("   - Use context managers for temporary tensors")
    print("   - Monitor memory usage during training")
    print("   - Use CPU for data preprocessing when possible")

def main():
    """Main validation function"""
    print("🔥 Starting GPU Validation for SSA Brain Tumor Segmentation")
    print("=" * 70)
    
    # Get GPU info
    gpu_available, gpu_memory, compute_cap = get_gpu_info()
    
    # Validate environment
    validate_environment()
    
    # Test GPU operations
    if gpu_available:
        test_success = test_gpu_operations()
        
        if test_success:
            print(f"\n🎉 GPU VALIDATION SUCCESSFUL!")
            print("=" * 60)
            print("✅ Your NVIDIA GeForce GTX 1650 is ready for:")
            print("   - 3D Brain Tumor Segmentation")
            print("   - Deep Learning Training")
            print("   - Intensive Medical Image Processing")
            
            # Provide recommendations
            recommend_optimizations(gpu_memory)
            
        else:
            print(f"\n⚠️ GPU validation failed - check CUDA installation")
    else:
        print(f"\n❌ GPU not available - will use CPU only (slower)")
    
    print(f"\n📋 NEXT STEPS:")
    print("=" * 60)
    print("1. 📊 Run SSA dataset analysis")
    print("2. 🔄 Execute GPU-optimized preprocessing")
    print("3. 🧠 Train 3D U-Net with GPU acceleration")
    print("4. 📈 Monitor GPU utilization during training")

if __name__ == "__main__":
    main()
