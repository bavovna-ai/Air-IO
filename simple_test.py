#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    import torch
    print("✅ PyTorch imported successfully")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from pyhocon import ConfigFactory
    print("✅ PyHocon imported successfully")
except ImportError as e:
    print(f"❌ PyHocon import failed: {e}")
    sys.exit(1)

try:
    from datasets.Bavovnadataset import Bavovna
    print("✅ Bavovna dataset imported successfully")
except ImportError as e:
    print(f"❌ Bavovna dataset import failed: {e}")
    sys.exit(1)

print("\nTesting dataset instantiation...")

try:
    # Test creating a Bavovna dataset instance
    dataset = Bavovna(
        data_root="/home/duarte33/Air-IO-og/data/Aurelia",
        data_name="1af74d402fd2e281-20000us",
        gravity=9.81007
    )
    print("✅ Bavovna dataset created successfully")
    print(f"Dataset length: {dataset.get_length()}")
    
    # Print some data info
    print(f"Time shape: {dataset.data['time'].shape}")
    print(f"Acc shape: {dataset.data['acc'].shape}")
    print(f"Gyro shape: {dataset.data['gyro'].shape}")
    print(f"GT orientation shape: {dataset.data['gt_orientation'].shape}")
    print(f"GT translation shape: {dataset.data['gt_translation'].shape}")
    
except Exception as e:
    print(f"❌ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed!") 