from typing import List, Dict, Tuple, Any, Optional, Union

import torch
from torch import Tensor

def imu_seq_collate(data: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    gt_rot = torch.stack([d["gt_rot"] for d in data])
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    init_rot = torch.stack([d["init_rot"] for d in data])
    init_vel = torch.stack([d["init_vel"] for d in data])

    dt = torch.stack([d["dt"] for d in data])

    return {
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "gt_rot": gt_rot,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_rot": init_rot,
    }


def custom_collate(data: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    dt = torch.stack([d["dt"] for d in data])
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])
    rot = torch.stack([d["rot"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    gt_rot = torch.stack([d["gt_rot"] for d in data])
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    init_rot = torch.stack([d["init_rot"] for d in data])
    init_vel = torch.stack([d["init_vel"] for d in data])

    return (
        {
            "dt": dt,
            "acc": acc,
            "gyro": gyro,
            "rot": rot,
        },
        {
            "pos": init_pos,
            "vel": init_vel,
            "rot": init_rot,
        },
        {
            "gt_pos": gt_pos,
            "gt_vel": gt_vel,
            "gt_rot": gt_rot,
        },
    )

def motion_collate_data(data: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Collate function that handles dynamic features from dataset configuration.
    
    Args:
        data: List of dictionaries containing tensors
        
    Returns:
        Tuple of dictionaries containing:
        - input_data: Features and metadata
        - init_state: Initial state values
        - label: Ground truth labels
    """
    # Initialize input data dict with timestamp if available
    input_data: Dict[str, Optional[Tensor]] = {}
    timestamp_list = [d['timestamp'] for d in data if 'timestamp' in d]
    if timestamp_list:
        input_data['ts'] = torch.stack(timestamp_list)
    
    # Add all available features from the first sample
    # This assumes all samples have the same features
    feature_keys = [k for k in data[0].keys() if not k.startswith(('init_', 'gt_'))]
    for key in feature_keys:
        if all(key in d for d in data):
            input_data[key] = torch.stack([d[key] for d in data])
    
    # Handle initial state
    init_keys = [k for k in data[0].keys() if k.startswith('init_')]
    init_state = {}
    for key in init_keys:
        base_key = key[5:]  # Remove 'init_' prefix
        if all(key in d for d in data):
            init_state[base_key] = torch.stack([d[key] for d in data])
    
    # Handle ground truth labels
    label_keys = [k for k in data[0].keys() if k.startswith('gt_')]
    label = {}
    for key in label_keys:
        base_key = key#[3:]  # Remove 'gt_' prefix
        if all(key in d for d in data):
            label[base_key] = torch.stack([d[key] for d in data])
    
    return input_data, init_state, label
    
def motion_collate(
    data: List[Dict[str, Tensor]], 
    **kwargs: Any
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    input_data, init_state, label = motion_collate_data(data)
    if kwargs:  # Changed from len(kwargs) > 0 for better Python style
        # TODO: Implement data augmentation if needed
        pass  
    return input_data, init_state, label

    
collate_fcs: Dict[str, Any] = {
    "base": custom_collate,
    'motion': motion_collate,
}
