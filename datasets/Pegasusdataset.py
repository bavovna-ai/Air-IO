import os
import logging
from typing import Dict, Any, Optional
import copy

import numpy as np
import torch
import pypose as pp

from .dataset import IMUSequence

logger = logging.getLogger(__name__)

class Pegasus(IMUSequence):
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
        self.load_imu(data_path)
        self.load_gt(data_path)

        # get the index for the data
        t_start = np.max([self.data["gt_time"][0], self.data["time"][0]])
        t_end = np.min([self.data["gt_time"][-1], self.data["time"][-1]])

        idx_start_imu = np.searchsorted(self.data["time"], t_start)
        idx_start_gt = np.searchsorted(self.data["gt_time"], t_start)

        idx_end_imu = np.searchsorted(self.data["time"], t_end, "right")
        idx_end_gt = np.searchsorted(self.data["gt_time"], t_end, "right")

        for k in ["gt_time", "pos", "quat","vel","body_vel"]:
            self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

        for k in ["time", "acc", "gyro"]:
            self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

        self.data["gt_translation"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["pos"]
        )
        self.data["velocity"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["vel"]
        )
        self.data["gt_orientation"] = self.interp_rot(
            self.data["time"], self.data["gt_time"], self.data["quat"]
        )

        # when evaluation: load airimu or integrated orientation:
        self.set_orientation(rot_path, data_name, rot_type)
        
        # transform to global/body frame:
        self.update_coordinate(coordinate, mode)
        
        # remove gravity term
        self.remove_gravity(remove_g)

    def load_imu(self, folder: str) -> None:
        """Load IMU data from CSV file."""
        imu_data = np.loadtxt(
            os.path.join(folder, "imu_data.csv"), dtype=float, delimiter=",", skiprows=1
        )
        #timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
        ## imu data: in FRD frame.
        self.data["time"] = torch.tensor(imu_data[:, 0], dtype=self.dtype)
        self.data["acc"] = torch.tensor(imu_data[:, 1:4], dtype=self.dtype)
        self.data["gyro"] = torch.tensor(imu_data[:, 4:], dtype=self.dtype)
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)
        
        # Rotation from FLU to FRD frame
        q_FLU_to_FRD = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.dtype)
        self.rot_FLU_to_FRD = pp.SO3(q_FLU_to_FRD)
        self.data["gyro"] = self.rot_FLU_to_FRD * self.data["gyro"]
        self.data["acc"] = self.rot_FLU_to_FRD * self.data["acc"]
        
    def load_gt(self, folder: str) -> None:
        """Load ground truth data from CSV file."""
        gt_data = np.loadtxt(
            os.path.join(folder, "ground_truth.csv"),
            dtype=float,
            delimiter=",",
            skiprows=1
        )
        #timestamp,q_x,q_y,q_z,q_w,v_x,v_y,v_z,b_v_x,b_v_y,b_v_z,p_x,p_y,p_z
        self.data["gt_time"] = torch.tensor(gt_data[:, 0], dtype=self.dtype)
        self.data["quat"] = torch.tensor(gt_data[:, 1:5], dtype=self.dtype) #x, y, z, w
        self.data["vel"] = torch.tensor(gt_data[:, 5:8], dtype=self.dtype)  #global_vel
        self.data["body_vel"] = torch.tensor(gt_data[:, 8:11], dtype=self.dtype) #body_vel
        self.data["pos"] = torch.tensor(gt_data[:, 11:14], dtype=self.dtype) #global_pos