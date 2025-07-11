import os
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import pypose as pp

from .dataset import IMUSequence

logger = logging.getLogger(__name__)

class EuRoC(IMUSequence):
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
        # load imu data
        self.load_imu(data_path)
        self.load_gt(data_path)
        # EUROC require an interpolation

        t_start = np.max([self.data["gt_time"][0], self.data["time"][0]])
        t_end = np.min([self.data["gt_time"][-1], self.data["time"][-1]])
        # find the index of the start and end
        idx_start_imu = np.searchsorted(self.data["time"], t_start)
        idx_start_gt = np.searchsorted(self.data["gt_time"], t_start)

        idx_end_imu = np.searchsorted(self.data["time"], t_end, "right")
        idx_end_gt = np.searchsorted(self.data["gt_time"], t_end, "right")

        for k in ["gt_time", "pos", "quat", "velocity"]:
            self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

        for k in ["time", "acc", "gyro"]:
            self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

        ## start interpotation
        self.data["gt_orientation"] = self.interp_rot(
            self.data["time"], self.data["gt_time"], self.data["quat"]
        )
        self.data["gt_translation"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["pos"]
        )

        self.data["velocity"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["velocity"]
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
            os.path.join(folder, "mav0/imu0/data.csv"), dtype=float, delimiter=","
        )
        self.data["time"] = torch.tensor(imu_data[:, 0] / 1e9, dtype=self.dtype)
        self.data["gyro"] = torch.tensor(imu_data[:, 1:4], dtype=self.dtype)  # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.data["acc"] = torch.tensor(imu_data[:, 4:], dtype=self.dtype)  # acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)
        
    def load_gt(self, folder: str) -> None:
        """Load ground truth data from CSV file."""
        gt_data = np.loadtxt(
            os.path.join(folder, "mav0/state_groundtruth_estimate0/data.csv"),
            dtype=float,
            delimiter=",",
        )
        self.data["gt_time"] = torch.tensor(gt_data[:, 0] / 1e9, dtype=self.dtype)
        self.data["pos"] = torch.tensor(gt_data[:, 1:4], dtype=self.dtype)
        self.data["quat"] = torch.tensor(gt_data[:, 4:8], dtype=self.dtype)  # w, x, y, z
        self.data["velocity"] = torch.tensor(gt_data[:, -9:-6], dtype=self.dtype)
