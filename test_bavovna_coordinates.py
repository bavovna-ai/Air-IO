#!/usr/bin/env python3
"""
Test script to verify the coordinate transformation in Bavovna dataset.
This script checks that the NED to world frame transformation is working correctly.
"""

import os
import sys
import numpy as np
import torch

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath('.')))

from datasets.Bavovnadataset import Bavovna

def test_coordinate_transformation():
    """Test the coordinate transformation from NED to world frame"""
    print("🧪 TESTING BAVOVNA COORDINATE TRANSFORMATION")
    print("=" * 60)
    
    # Load the dataset
    dataset = Bavovna(
        data_root="/home/duarte33/AirIO-Bavovna/data/Aurelia",
        data_name="1af74d402fd2e281-20000us.csv",
        mode="train",
        rot_path=None,
        rot_type=None,
    )
    
    # Run verification
    dataset.verify_coordinate_transformation()
    
    # Additional tests
    print("\n🔬 ADDITIONAL COORDINATE TESTS")
    print("-" * 40)
    
    # Test 1: Check gravity direction
    acc_mean = torch.mean(dataset.data["acc"], dim=0)
    gravity_magnitude = torch.norm(acc_mean).item()
    print(f"Gravity magnitude: {gravity_magnitude:.3f} m/s²")
    print(f"Expected: ~9.81 m/s²")
    
    # Test 2: Check Z-axis orientation
    pos = dataset.data["gt_translation"]
    z_positive_ratio = torch.sum(pos[:, 2] > 0).item() / len(pos) * 100
    print(f"Z-axis positive ratio: {z_positive_ratio:.1f}%")
    print(f"Expected: >50% (most positions should be above ground)")
    
    # Test 3: Check velocity consistency
    vel = dataset.data["velocity"]
    vel_magnitude = torch.norm(vel, dim=1)
    max_vel = torch.max(vel_magnitude).item()
    print(f"Maximum velocity: {max_vel:.3f} m/s")
    print(f"Expected: Reasonable drone speed (<50 m/s)")
    
    # Test 4: Check quaternion validity
    if 'gt_orientation' in dataset.data:
        quat = dataset.data["gt_orientation"]
        if hasattr(quat, 'tensor'):
            quat_data = quat.tensor()
        else:
            quat_data = quat
        
        quat_norms = torch.norm(quat_data, dim=1)
        norm_valid = torch.all(torch.abs(quat_norms - 1.0) < 0.01)
        print(f"Quaternions normalized: {norm_valid.item()}")
        print(f"Expected: True")
    
    print("\n✅ Coordinate transformation test complete!")

if __name__ == "__main__":
    test_coordinate_transformation() 