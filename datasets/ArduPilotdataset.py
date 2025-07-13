"""
ArduPilot dataset reader for CSV data.
"""
import os
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch
import pypose as pp
from scipy.spatial.transform import Rotation

from .dataset import IMUSequence

logger = logging.getLogger(__name__)

class ArduPilot(IMUSequence):
    """
    ArduPilot dataset reader for CSV data with columns:
    TimeUS, AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ, Alt, Roll, Pitch, Yaw,
    PN, VN, PE, VE, PD, VD, Lat, Lng, Status
    """
    feature_dict = {
        "acc": ["AccX", "AccY", "AccZ"],
        "gyro": ["GyrX", "GyrY", "GyrZ"],
        "Mag": ["MagX", "MagY", "MagZ"],
        "Alt": ["Alt"]
    }

    def __init__(
        self,
        data_root: str,
        data_name: str,
        coordinate: Optional[str] = None,
        mode: Optional[str] = None,
        rot_path: Optional[str] = None,
        rot_type: Optional[str] = None,
        gravity: float = 9.81007,
        remove_g: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            data_root=data_root,
            data_name=data_name,
            coordinate=coordinate,
            mode=mode,
            rot_path=rot_path,
            rot_type=rot_type,
            gravity=gravity,
            remove_g=remove_g,
            **kwargs
        )
        
        data_path = os.path.join(data_root, data_name)
        logger.info(f"Loading ArduPilot sequence from {data_path}")
        
        # Load IMU and ground truth data
        self.load_imu(data_path)
        self.load_gt(data_path)

        # Find common time range
        t_start = np.max([self.data["gt_time"][0], self.data["time"][0]])
        t_end = np.min([self.data["gt_time"][-1], self.data["time"][-1]])
        
        # Find indices for the common time range
        idx_start_imu = np.searchsorted(self.data["time"].numpy(), t_start)
        idx_start_gt = np.searchsorted(self.data["gt_time"].numpy(), t_start)
        idx_end_imu = np.searchsorted(self.data["time"].numpy(), t_end, "right")
        idx_end_gt = np.searchsorted(self.data["gt_time"].numpy(), t_end, "right")
        
        # Trim data to common time range
        for k in ["gt_time", "pos", "quat", "velocity"]:
            self.data[k] = self.data[k][idx_start_gt:idx_end_gt]
        
        for k in ["time"] + list(self.feature_names):
            if k in self.data:
                self.data[k] = self.data[k][idx_start_imu:idx_end_imu]
        
        # Interpolate ground truth to IMU timestamps
        self.data["gt_orientation"] = self.interp_rot(
            self.data["time"], self.data["gt_time"], self.data["quat"]
        )
        self.data["gt_translation"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["pos"]
        )
        self.data["velocity"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["velocity"]
        )
        
        # Calculate time differences
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

        # Set up orientation and coordinate frame
        self.set_orientation(rot_path, data_name, rot_type)
        self.update_coordinate(coordinate, mode)
        
        # Remove gravity if requested
        self.remove_gravity(remove_g)

    def get_feature_config(self) -> Dict[str, List[str]]:
        """
        Get the feature configuration for ArduPilot dataset.
        The features are derived from CSV columns:
        TimeUS, AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ, Alt, Roll, Pitch, Yaw,
        PN, VN, PE, VE, PD, VD, Lat, Lng, Status
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping feature groups to their column names
        """
        return {
            "acc": ["AccX", "AccY", "AccZ"],
            "gyro": ["GyrX", "GyrY", "GyrZ"],
            "Mag": ["MagX", "MagY", "MagZ"],
            "Alt": ["Alt"]
        }

    def _parse_config(self) -> None:
        # Get feature configuration
        features_config = self.conf.get("features", {})
        if not features_config:
            raise ValueError("No features defined in configuration")
        
        # Store feature structure from config
        self.feature_groups = list(features_config.keys())  # Preserve config order
        self.feature_dict = features_config  # Direct mapping from config
        
        # Create flat list of all feature column names
        self.feature_names = []
        for columns in features_config.values():
            self.feature_names.extend(columns)
        
    def load_imu(self, csv_path: str) -> None:
        """Load IMU data and additional features from CSV file based on config."""
        self._parse_config()
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Reading data from {csv_path}")
            
            # Convert timestamps to seconds and to tensor
            self.data["time"] = torch.tensor(df["TimeUS"].values / 1e6, dtype=self.dtype)  # microseconds to seconds
            
            # Load all features from config
            for group_name, columns in self.feature_dict.items():
                if not all(col in df.columns for col in columns):
                    logger.warning(f"Missing columns for feature group {group_name}: {columns}")
                    continue
                    
                logger.info(f"Loading {group_name} data: {columns}")
                self.data[group_name] = torch.tensor(
                    df[columns].values if len(columns) > 1 else df[columns[0]].values[:, None],
                    dtype=self.dtype
                )
            
            # Validate required IMU features
            if not all(f in self.data for f in ["acc", "gyro"]):
                raise ValueError("Required IMU features (acc, gyro) not found in data")
            
            # Log loaded features
            logger.info("Loaded feature groups: " + ", ".join(self.data.keys()))
            
        except Exception as e:
            logger.error(f"Failed to load data from {csv_path}: {str(e)}")
            raise

    def load_gt(self, csv_path: str) -> None:
        """Load ground truth data from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Reading ground truth data from {csv_path}")
            
            # Convert timestamps to seconds and to tensor
            self.data["gt_time"] = torch.tensor(df["TimeUS"].values / 1e6, dtype=self.dtype)  # microseconds to seconds
            
            # Ground truth position (NED frame)
            self.data["pos"] = torch.tensor(np.column_stack([
                df["PN"].values,  # North
                df["PE"].values,  # East
                -df["PD"].values  # Down (negative for NED->ENU conversion)
            ]), dtype=self.dtype)
            
            # Ground truth velocity (NED frame)
            self.data["velocity"] = torch.tensor(np.column_stack([
                df["VN"].values,  # North
                df["VE"].values,  # East
                -df["VD"].values  # Down (negative for NED->ENU conversion)
            ]), dtype=self.dtype)
            
            # Check if quaternion data is available
            quat_cols = ['Q1', 'Q2', 'Q3', 'Q4']
            if all(col in df.columns for col in quat_cols):
                # ArduPilot quaternions: Q1-Q4 = (w, x, y, z), already in correct order
                logger.info("Using quaternion data from log")
                self.data["quat"] = torch.tensor(df[quat_cols].values, dtype=self.dtype)
            else:
                # Convert Euler angles to quaternions
                logger.info("Converting Euler angles to quaternions")
                roll = df["Roll"].values * np.pi / 180.0  # deg to rad
                pitch = df["Pitch"].values * np.pi / 180.0
                yaw = df["Yaw"].values * np.pi / 180.0
                
                # Create rotation objects and convert to quaternions (w,x,y,z)
                rot = Rotation.from_euler('xyz', np.column_stack([roll, pitch, yaw]))
                quat = rot.as_quat()
                
                # Reorder from (x,y,z,w) to (w,x,y,z) and convert to tensor
                self.data["quat"] = torch.tensor(np.column_stack([
                    quat[:, 3],  # w
                    quat[:, 0],  # x
                    quat[:, 1],  # y
                    quat[:, 2]   # z
                ]), dtype=self.dtype)
            
        except Exception as e:
            logger.error(f"Failed to load ground truth data from {csv_path}: {str(e)}")
            raise 

    @property
    def feature_dim(self) -> int:
        """
        Total number of input features based on loaded data.
        
        Returns:
            int: Total number of input features
        """
        return len(self.feature_names)


if __name__ == "__main__":
    import argparse
    from pyhocon import ConfigFactory
    
    # Example config
    example_config = """
    features {
        acc = ["AccX", "AccY", "AccZ"]
        gyro = ["GyrX", "GyrY", "GyrZ"]
        Mag = ["MagX", "MagY", "MagZ"]
        Alt = ["Alt"]
    }
    """
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="ArduPilot Dataset Example")
    parser.add_argument("--data_path", type=str, default="data/sample.csv", help="Path to CSV data file")
    args = parser.parse_args()
    
    # Create dataset instance
    conf = ConfigFactory.parse_string(example_config)
    dataset = ArduPilot(
        data_root=".",
        data_name=args.data_path,
        conf=conf
    )
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Feature groups: {dataset.feature_groups}")
    print(f"Feature dictionary:")
    for group, columns in dataset.feature_dict.items():
        print(f"  {group}: {columns}")
    print(f"All feature names: {dataset.feature_names}")
    print(f"Total feature dimension: {dataset.feature_dim}")
    
    # Print data shapes
    print("\nLoaded Data Shapes:")
    for group in dataset.feature_groups:
        if group in dataset.data:
            print(f"  {group}: {dataset.data[group].shape}")