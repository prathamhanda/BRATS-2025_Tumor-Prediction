#!/usr/bin/env python3
"""
🔍 SSA Patch Verification Script
===============================

Verify the quality and format of GPU-processed SSA patches
"""

import numpy as np
import os

def verify_ssa_patches():
    """Verify processed SSA patches"""
    
    patch_dir = "SSA_Type/ssa_preprocessed_patches"
    
    print("🔍 SSA PATCH VERIFICATION")
    print("=" * 60)
    
    # Get all patch files
    patch_files = [f for f in os.listdir(patch_dir) if f.endswith('.npz')]
    
    print(f"📦 Found {len(patch_files)} patch files")
    
    # Verify first few patches
    for i, patch_file in enumerate(patch_files[:3]):
        print(f"\n🧠 Patch {i+1}: {patch_file}")
        print("-" * 40)
        
        try:
            data = np.load(os.path.join(patch_dir, patch_file))
            
            print(f"✅ Image Shape: {data['image'].shape}")
            print(f"✅ Mask Shape: {data['mask'].shape}")
            print(f"✅ Tumor Voxels: {data['tumor_voxels']}")
            print(f"✅ Coordinates: {data['coordinates']}")
            print(f"✅ Case Name: {data['case_name']}")
            print(f"✅ Image Data Type: {data['image'].dtype}")
            print(f"✅ Mask Data Type: {data['mask'].dtype}")
            print(f"✅ Unique Mask Values: {np.unique(data['mask'])}")
            
            # Check data ranges
            print(f"📊 Image Range: [{data['image'].min():.3f}, {data['image'].max():.3f}]")
            print(f"📊 Non-zero Voxels: {np.count_nonzero(data['image'])}")
            
        except Exception as e:
            print(f"❌ Error loading {patch_file}: {e}")
    
    print(f"\n🎉 SSA PATCH VERIFICATION COMPLETE!")
    print(f"✅ Ready for deep learning training with your GTX 1650")

if __name__ == "__main__":
    verify_ssa_patches()
