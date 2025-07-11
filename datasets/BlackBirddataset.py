"""
Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/learning/data_management/prepare_datasets/blackbird.py
"""
import os
import logging
from typing import Dict, Any, Optional
import copy

import numpy as np
import torch
import pypose as pp
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d

from .dataset import IMUSequence

logger = logging.getLogger(__name__)

class BlackBird(IMUSequence):
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
        self.load_imu(data_path, data_name)
        self.load_gt(data_path)
        self.refer_IMO()
        
        # when evaluation: load airimu or integrated orientation:
        self.set_orientation(rot_path, data_name, rot_type)
        
        # transform to global/body frame:
        self.update_coordinate(coordinate, mode)
        
        # remove gravity term
        self.remove_gravity(remove_g)

    def refer_IMO(self) -> None:
        """Transform data to IMO reference frame."""
        # the provided ground truth is the drone body in the NED vicon frame
        # rotate to have z upwards
        R_w_ned = np.array([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.]])
        t_w_ned = np.array([0., 0., 0.])

        # rotate from body to imu frame
        R_b_i = np.array([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 1.]])
        t_b_i = np.array([0., 0., 0.])
        
        raw_imu = np.asarray(self.imu_data)
        thrusts = np.asarray(self.thrusts)
        data_tmp = self.gt_data

        data = []
        for data_i in data_tmp:
            ts_i = data_i[0] / 1e6
            
            t_i = data_i[1:4]
            R_i = Rotation.from_quat(
                np.array([data_i[5], data_i[6], data_i[7], data_i[4]])).as_matrix()

            # transform to world frame
            R_it = R_w_ned @ R_i
            t_it = t_w_ned + R_w_ned @ t_i

            # transform to imu frame
            t_it = t_it + R_it @ t_b_i
            R_it = R_it @ R_b_i

            q_it = Rotation.from_matrix(R_it).as_quat()
            d = np.array([
                ts_i,
                t_it[0], t_it[1], t_it[2],
                q_it[0], q_it[1], q_it[2], q_it[3]
            ])
            data.append(d)
        data = np.asarray(data)

        # include velocities
        gt_times = data[:, 0] 
        gt_pos = data[:, 1:4]

        # compute velocity
        v_start = ((gt_pos[1] - gt_pos[0]) / (gt_times[1] - gt_times[0])).reshape((1, 3))
        gt_vel_raw = (gt_pos[1:] - gt_pos[:-1]) / (gt_times[1:] - gt_times[:-1])[:, None]
        gt_vel_raw = np.concatenate((v_start, gt_vel_raw), axis=0)
        # filter
        gt_vel_x = np.convolve(gt_vel_raw[:, 0], np.ones(5) / 5, mode='same')
        gt_vel_x = gt_vel_x.reshape((-1, 1))
        gt_vel_y = np.convolve(gt_vel_raw[:, 1], np.ones(5) / 5, mode='same')
        gt_vel_y = gt_vel_y.reshape((-1, 1))
        gt_vel_z = np.convolve(gt_vel_raw[:, 2], np.ones(5) / 5, mode='same')
        gt_vel_z = gt_vel_z.reshape((-1, 1))
        gt_vel = np.concatenate((gt_vel_x, gt_vel_y, gt_vel_z), axis=1)

        gt_traj_tmp = np.concatenate((data, gt_vel), axis=1)  # [ts x y z qx qy qz qw vx vy vz]

        # In Blackbird dataset, the sensors measurements are at:
        # 100 Hz IMU meas.
        # 180 Hz RPM meas.
        # 360 Hz Vicon meas.
        # resample imu at exactly 100 Hz
        t_curr = raw_imu[0, 0]
        dt = 0.01
        new_times_imu = [t_curr]
        while t_curr < raw_imu[-1, 0] - dt - 0.001:
            t_curr = t_curr + dt
            new_times_imu.append(t_curr)
        new_times_imu = np.asarray(new_times_imu)
        gyro_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 1:4], axis=0)(new_times_imu)
        accel_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 4:7], axis=0)(new_times_imu)
        raw_imu = np.concatenate((new_times_imu.reshape((-1, 1)), gyro_tmp, accel_tmp), axis=1)

        # We down sample to IMU rate
        times_imu = raw_imu[:, 0]
        # get initial and final times for interpolations
        idx_s = 0
       
        for ts in times_imu:
            if ts > gt_traj_tmp[0, 0] and ts > thrusts[0, 0]:
                break
            else:
                idx_s = idx_s + 1
        assert idx_s < len(times_imu)

        idx_e = len(times_imu) - 1
        for ts in reversed(times_imu):
            if ts < gt_traj_tmp[-1, 0] and ts < thrusts[-1, 0]:
                break
            else:
                idx_e = idx_e - 1
        assert idx_e > 0

        idx_e = len(times_imu) - 1
        for ts in reversed(times_imu):
            if ts < gt_traj_tmp[-1, 0]:
                break
            else:
                idx_e = idx_e - 1
        assert idx_e > 0

        times_imu = times_imu[idx_s:idx_e + 1]
 
        raw_imu = raw_imu[idx_s:idx_e + 1]
        self.data["gyro"] = torch.tensor(raw_imu[:, 1:4], dtype=self.dtype)
        self.data["acc"] = torch.tensor(raw_imu[:, 4:], dtype=self.dtype)

        # interpolate ground-truth samples at thrust times
        self.data["gt_translation"] = torch.tensor(
            interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 1:4], axis=0)(times_imu),
            dtype=self.dtype
        )
        # alternative to Slerp (from scipy.spatial.transform import Slerp)
        self.data["gt_orientation"] = pp.SO3(torch.tensor(
            Rotation.from_quat(gt_traj_tmp[:, 4:8])(times_imu).as_quat(),
            dtype=self.dtype
        ))
        self.data["velocity"] = torch.tensor(
            interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 8:11], axis=0)(times_imu),
            dtype=self.dtype
        )
        
        self.data["gt_time"] = torch.tensor(times_imu, dtype=self.dtype)
        self.data["time"] = torch.tensor(times_imu, dtype=self.dtype)
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool, dtype=self.dtype)

    def load_imu(self, folder: str, data_name: Optional[str] = None) -> None:
        """Load IMU data from CSV file."""
        imu_data = np.loadtxt(
            os.path.join(folder, "imu_data.csv"), dtype=float, delimiter=","
        )
        thrusts = np.loadtxt(
            os.path.join(folder, "thrust_data.csv"), dtype=float, delimiter=","
        )
        self.imu_data = copy.deepcopy(imu_data)
        self.thrusts = copy.deepcopy(thrusts)

    def load_gt(self, folder: str) -> None:
        """Load ground truth data from CSV file."""
        gt_data = np.loadtxt(
            os.path.join(folder, "groundTruthPoses.csv"),
            dtype=float,
            delimiter=",",
        )
        self.gt_data = copy.deepcopy(gt_data)