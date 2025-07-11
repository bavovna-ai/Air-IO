import os
import logging
from typing import Dict, Any, Optional

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
        
        for k in ["time", "acc", "gyro"]:
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

    def load_imu(self, csv_path: str) -> None:
        """Load IMU data from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Reading IMU data from {csv_path}")
            
            # Convert timestamps to seconds and to tensor
            self.data["time"] = torch.tensor(df["TimeUS"].values / 1e6, dtype=self.dtype)  # microseconds to seconds
            
            # IMU data - acceleration in m/s^2, gyro in rad/s
            self.data["gyro"] = torch.tensor(np.column_stack([
                df["GyrX"].values,
                df["GyrY"].values,
                df["GyrZ"].values
            ]), dtype=self.dtype)
            
            self.data["acc"] = torch.tensor(np.column_stack([
                df["AccX"].values,
                df["AccY"].values,
                df["AccZ"].values
            ]), dtype=self.dtype)
            
        except Exception as e:
            logger.error(f"Failed to load IMU data from {csv_path}: {str(e)}")
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