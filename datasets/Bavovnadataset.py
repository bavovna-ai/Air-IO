import os
import numpy as np
import pypose as pp
import torch
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import lookAt, qinterp, Gaussian_noise
from .dataset import Sequence
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

class Bavovna(Sequence):
    """
    ArduPilot dataset reader for CSV data with columns:
    TimeUS, AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ, Alt, Roll, Pitch, Yaw,
    PN, VN, PE, VE, PD, VD, Lat, Lng, Status
    """
    def __init__(self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.81007, 
        remove_g=False,
        **kwargs
    ):
        super(Bavovna, self).__init__()
        (
            self.data_root, 
            self.data_name,
            self.data, # dictionary to store the data
            self.ts, # time stamp
            self.targets, # targets
            self.orientations, # orientations
            self.gt_pos, # ground truth position
            self.gt_ori, # ground truth orientation
            self.coordinate, # coordinate   
            self.mode, # mode
        ) = (data_root, data_name, dict(), None, None, None, None, None, coordinate, mode)
        print("coordinate: ", self.coordinate)
        print("mode: ", self.mode)
        # Gravity vector in world frame (Z-up)
        # After NED to world transformation, gravity points downward (-Z)
        self.g_vector = torch.tensor([0, 0, -gravity], dtype=torch.double)
        self.load_data(data_root, data_name) # For Bavovna dataset, the CSV file is directly in the data_root directory
        self.convert_to_torch() # Convert to torch tensors
        self.set_orientation(rot_path, data_name, rot_type) # For Bavovna dataset, we can leave it empty as we're using ground truth orientation
        self.update_coordinate(coordinate, mode) # For Bavovna dataset, we can leave it empty as we're using ground truth orientation
        self.remove_gravity(remove_g) # Remove gravity if needed
    
    def get_length(self): # Get the length of the dataset
        return self.data["time"].shape[0]
    
    def load_data(self, folder, file_name):        
        # Load the CSV file
        csv_file = os.path.join(folder, file_name)  # Assuming your file is named data.csv
        if not os.path.exists(csv_file):
            # Try alternative names
            for filename in ["imu_data.csv", "flight_data.csv", "bavovna_data.csv"]:
                if os.path.exists(os.path.join(folder, filename)):
                    csv_file = os.path.join(folder, filename)
                    break
        # Load data with pandas for better column handling
        df = pd.read_csv(csv_file)
        
        # Extract IMU data in body frame
        self.data["time"] = df["TimeUS"].values / 1e6  # Convert microseconds to seconds
        acc_raw = df[["AccX", "AccY", "AccZ"]].values
        gyro_raw = df[["GyrX", "GyrY", "GyrZ"]].values
        self.data["acc"] = np.array([acc_raw[:, 0], acc_raw[:, 1], acc_raw[:, 2]]).T
        self.data["gyro"] = np.array([gyro_raw[:, 0], gyro_raw[:, 1], gyro_raw[:, 2]]).T

        # Extract ground truth data from NED which is a type of world/global frame
        self.data["gt_time"] = df["TimeUS"].values / 1e6  # Convert microseconds to seconds
        # Transform position and velocity from NED to ENU frame
        pos_ned = df[["PN", "PE", "PD"]].values
        vel_ned = df[["VN", "VE", "VD"]].values 
        self.data["pos"] = np.array([pos_ned[:, 0], pos_ned[:, 1], -pos_ned[:, 2]]).T  # North, East, Up
        self.data["velocity"] = np.array([vel_ned[:, 0], vel_ned[:, 1], -vel_ned[:, 2]]).T  # North, East, Up velocity
        # Quaternion transformation for NED to world frame
        # For NED to world (Z-up), we need to rotate around X-axis by 180 degrees, this is equivalent to negating the Y and Z components of the quaternion
        quat_raw = df[["Q1", "Q2", "Q3", "Q4"]].values # Load raw quaternion (w, x, y, z)
        quat_xyzw = quat_raw[:, [1, 2, 3, 0]]# Convert to (x, y, z, w) for scipy
        r_ned_to_enu = R.from_euler('x', 180, degrees=True) # Apply 180° rotation about X-axis to convert NED to ENU/world frame
        r_body_ned = R.from_quat(quat_xyzw)  # Rotate from NED to ENU/world frame
        r_body_world = r_ned_to_enu * r_body_ned  # Compose the rotations
        quat_world = r_body_world.as_quat()[:, [3, 0, 1, 2]] # Convert back to (w, x, y, z)
        self.data["quat"] = quat_world

        # Additional data that might be useful
        self.data["altitude"] = df["Alt"].values
        self.data["magnetometer"] = df[["MagX", "MagY", "MagZ"]].values
        self.data["euler"] = df[["Roll", "Pitch", "Yaw"]].values
        self.data["gps"] = df[["Lat", "Lng"]].values
        self.data["status"] = df["Status"].values
        
        # Ensure data synchronization (even if timestamps are the same)
        # This is important for consistency with other datasets
        t_start = np.max([self.data["gt_time"][0], self.data["time"][0]])
        t_end = np.min([self.data["gt_time"][-1], self.data["time"][-1]])
        
        # Find the index of the start and end
        idx_start_imu = np.searchsorted(self.data["time"], t_start)
        idx_start_gt = np.searchsorted(self.data["gt_time"], t_start)
        idx_end_imu = np.searchsorted(self.data["time"], t_end, "right")
        idx_end_gt = np.searchsorted(self.data["gt_time"], t_end, "right")
        
        # Trim data to synchronized range
        for k in ["gt_time", "pos", "quat", "velocity"]:
            self.data[k] = self.data[k][idx_start_gt:idx_end_gt]
        for k in ["time", "acc", "gyro"]:
            self.data[k] = self.data[k][idx_start_imu:idx_end_imu]
    
        # Interpolate GT data to IMU timestamps for consistency
        self.data["gt_orientation"] = self.interp_rot(self.data["time"], self.data["gt_time"], self.data["quat"])
        self.data["gt_translation"] = self.interp_xyz(self.data["time"], self.data["gt_time"], self.data["pos"])
        self.data["velocity"] = self.interp_xyz(self.data["time"], self.data["gt_time"], self.data["velocity"])

        self.update_coordinate(self.coordinate, self.mode) # Update coordinate to body frame
    
    def interp_rot(self, time, opt_time, quat):
        """Interpolate quaternion data to IMU timestamps"""
        print("  Starting quaternion interpolation...")
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])
        quat = torch.tensor(quat)
        print(f"  Calling qinterp with shapes: quat={quat.shape}, gt_dt={gt_dt.shape}, imu_dt={imu_dt.shape}")
        quat = qinterp(quat, gt_dt, imu_dt).double()
        print("  qinterp completed")
        self.data["rot_wxyz"] = quat
        rot = torch.zeros_like(quat)
        rot[:, 3] = quat[:, 0]
        rot[:, :3] = quat[:, 1:]
        print(f"  Creating SO3 object with shape: {rot.shape}")
        return pp.SO3(rot)

    def interp_xyz(self, time, opt_time, xyz):
        """Interpolate position/velocity data to IMU timestamps"""
        intep_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
        intep_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
        intep_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])
        
        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)
    
    def convert_to_torch(self):
        """Convert numpy arrays to torch tensors"""
        self.data["time"] = torch.tensor(self.data["time"]).double()
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None].double()
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)
        self.data["gyro"] = torch.tensor(self.data["gyro"]).double()
        self.data["acc"] = torch.tensor(self.data["acc"]).double()
        
        # Convert GT data
        self.data["gt_orientation"] = self.data["gt_orientation"].double()
        self.data["gt_translation"] = torch.tensor(self.data["gt_translation"]).double()
        self.data["velocity"] = torch.tensor(self.data["velocity"]).double()
        
        # Convert additional data
        self.data["altitude"] = torch.tensor(self.data["altitude"]).double()
        self.data["magnetometer"] = torch.tensor(self.data["magnetometer"]).double()
        self.data["euler"] = torch.tensor(self.data["euler"]).double()
        self.data["gps"] = torch.tensor(self.data["gps"]).double()
    
    def update_coordinate(self, coordinate, mode):
        """Transform between coordinate frames"""
        if coordinate == "glob_coord": #Rotates the IMU signals (gyro and acc) from the body frame to the global frame.
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"] 
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
        elif coordinate == "body_coord": 
                print("updating coordinate to body frame\n")
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode != "infevaluate" and mode != "inference":
                    self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
    
    def set_orientation(self, exp_path, data_name, rotation_type):
        """Set orientation for evaluation (optional)"""
        # This method is used for evaluation when loading pre-trained orientation
        # For Bavovna, we can leave it empty as we're using ground truth orientation
        pass
    
    def remove_gravity(self, remove_g):
        """Remove gravity from accelerometer data"""
        if remove_g:
            # Remove gravity component from accelerometer data
            # This depends on your coordinate frame and gravity direction+
            print("removing gravity\n")
            gravity_vector = self.g_vector.unsqueeze(0).expand(self.data["acc"].shape[0], -1)
            self.data["acc"] = self.data["acc"] - gravity_vector


if __name__ == "__main__":
    dataset = Bavovna(
        data_root = "/home/duarte33/AirIO-Bavovna/data/Aurelia",
        data_name = "1af74d402fd2e281-20000us.csv",
        mode="train",
        rot_path=None,
        rot_type=None,
    )
    
    # Verify coordinate transformation
    dataset.verify_coordinate_transformation()
    
    print("\nDATASET SUMMARY")
    print("=" * 30)
    print(f"Time range: {dataset.data['time'][0]:.2f} - {dataset.data['time'][-1]:.2f} s")
    print(f"Total frames: {dataset.get_length()}")
    print(f"Frame rate: {dataset.get_length() / (dataset.data['time'][-1] - dataset.data['time'][0]):.1f} Hz")
    print(f"Position range: X=[{torch.min(dataset.data['gt_translation'][:, 0]):.1f}, {torch.max(dataset.data['gt_translation'][:, 0]):.1f}] m")
    print(f"                Y=[{torch.min(dataset.data['gt_translation'][:, 1]):.1f}, {torch.max(dataset.data['gt_translation'][:, 1]):.1f}] m")
    print(f"                Z=[{torch.min(dataset.data['gt_translation'][:, 2]):.1f}, {torch.max(dataset.data['gt_translation'][:, 2]):.1f}] m")
