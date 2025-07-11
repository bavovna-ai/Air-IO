import argparse
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Union

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory
import pypose as pp
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import pickle
import os
from utils import qinterp

logger = logging.getLogger(__name__)


class Sequence(ABC):
    """
    An abstract base class for sequence data.
    """
    # Dictionary to keep track of subclasses
    subclasses: Dict[str, Type['Sequence']] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Registers subclasses of the Sequence class.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, dtype = torch.double) -> None:
        """
        Initialize the sequence with empty data.
        """
        self.dtype = dtype
        self.data: Dict[str, Any] = {
            "time": torch.tensor([]),
            "dt": torch.tensor([]),
            "acc": torch.tensor([]),
            "gyro": torch.tensor([]),
            "gt_orientation": pp.SO3(torch.tensor([[0., 0., 0., 1.]], dtype=self.dtype)),
            "gt_translation": torch.tensor([]),
            "velocity": torch.tensor([])
        }

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the sequence.
        """
        raise NotImplementedError

    def interp_rot(self, time: np.ndarray, opt_time: np.ndarray, quat: np.ndarray) -> pp.SO3:
        """
        Interpolate rotation data.
        
        Args:
            time: Target timestamps
            opt_time: Original timestamps
            quat: Quaternion data in [x, y, z, w] format
        
        Returns:
            Interpolated rotations as SO3 object
        """
        quat_wxyz = np.zeros_like(quat)
        quat_wxyz[:, 0] = quat[:, 3]
        quat_wxyz[:, 1:] = quat[:, :3]
        quat_wxyz = torch.tensor(quat_wxyz)
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])

        quat = qinterp(quat_wxyz, gt_dt, imu_dt).double()
        quat_xyzw = torch.zeros_like(quat)
        quat_xyzw[:, 3] = quat[:, 0]
        quat_xyzw[:, :3] = quat[:, 1:]
        return pp.SO3(quat_xyzw)

    def interp_xyz(self, time: np.ndarray, opt_time: np.ndarray, xyz: np.ndarray) -> torch.Tensor:
        """
        Interpolate position or velocity data.
        
        Args:
            time: Target timestamps
            opt_time: Original timestamps
            xyz: Position or velocity data
            
        Returns:
            Interpolated data as tensor
        """
        intep_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
        intep_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
        intep_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])

        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)

    def update_coordinate(self, coordinate: Optional[str], mode: Optional[str]) -> None:
        """
        Updates the data (imu / velocity) based on the required mode.
        
        Args:
            coordinate: The target coordinate system ('glob_coord' or 'body_coord')
            mode: The dataset mode, only rotating the velocity during training
        """
        if coordinate is None:
            logger.info("No coordinate system provided. Skipping update.")
            return
        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode != "infevaluate" and mode != "inference":
                    self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            logger.error("An error occurred while updating coordinates: %s", e)
            raise e

    def set_orientation(self, exp_path: Optional[str], data_name: str, rotation_type: Optional[str]) -> None:
        """
        Sets the ground truth orientation based on the provided rotation.
        
        Args:
            exp_path: Path to the pickle file containing orientation data
            data_name: Name of the data sequence
            rotation_type: The type of rotation (AirIMU corrected orientation / raw imu Pre-integration)
        """
        if rotation_type is None or rotation_type == "None" or rotation_type.lower() == "gtrot":
            return
        try:
            with open(exp_path, 'rb') as file:
                loaded_data = pickle.load(file)

            state = loaded_data[data_name]

            if rotation_type.lower() == "airimu":
                self.data["gt_orientation"] = state['airimu_rot']
            elif rotation_type.lower() == "integration":
                self.data["gt_orientation"] = state['inte_rot']
            else:
                logger.error("Unsupported rotation type: %s", rotation_type)
                raise ValueError(f"Unsupported rotation type: {rotation_type}")
        except FileNotFoundError:
            logger.error("The file %s was not found.", exp_path)
            raise

    def remove_gravity(self, remove_g: bool) -> None:
        """
        Remove gravity from accelerometer measurements.
        
        Args:
            remove_g: Whether to remove gravity
        """
        if remove_g is True:
            logger.info("Gravity has been removed")
            self.data["acc"] -= self.g_vector

    @abstractmethod
    def load_imu(self, folder: str) -> None:
        """
        Load IMU data from files.
        
        Args:
            folder: Path to the data folder
        """
        raise NotImplementedError

    @abstractmethod
    def load_gt(self, folder: str) -> None:
        """
        Load ground truth data from files.
        
        Args:
            folder: Path to the data folder
        """
        raise NotImplementedError


class IMUSequence(Sequence):
    """
    An intermediate class for IMU sequence data with common functionality.
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
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.dtype = torch.double
        self.g_vector = torch.tensor([0, 0, gravity], dtype=self.dtype)

    def __len__(self) -> int:
        """Returns the length of the sequence."""
        return len(self.data["time"])

    def interp_rot(self, time: np.ndarray, opt_time: np.ndarray, quat: np.ndarray) -> pp.SO3:
        """
        Interpolate rotation data.
        
        Args:
            time: Target timestamps
            opt_time: Original timestamps
            quat: Quaternion data in [x, y, z, w] format
        
        Returns:
            Interpolated rotations as SO3 object
        """
        quat_wxyz = np.zeros_like(quat)
        quat_wxyz[:, 0] = quat[:, 3]
        quat_wxyz[:, 1:] = quat[:, :3]
        quat_wxyz = torch.tensor(quat_wxyz)
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])

        quat = qinterp(quat_wxyz, gt_dt, imu_dt).double()
        quat_xyzw = torch.zeros_like(quat)
        quat_xyzw[:, 3] = quat[:, 0]
        quat_xyzw[:, :3] = quat[:, 1:]
        return pp.SO3(quat_xyzw)

    def interp_xyz(self, time: np.ndarray, opt_time: np.ndarray, xyz: np.ndarray) -> torch.Tensor:
        """
        Interpolate position or velocity data.
        
        Args:
            time: Target timestamps
            opt_time: Original timestamps
            xyz: Position or velocity data
            
        Returns:
            Interpolated data as tensor
        """
        intep_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
        intep_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
        intep_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])

        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)

    def update_coordinate(self, coordinate: Optional[str], mode: Optional[str]) -> None:
        """
        Updates the data (imu / velocity) based on the required mode.
        
        Args:
            coordinate: The target coordinate system ('glob_coord' or 'body_coord')
            mode: The dataset mode, only rotating the velocity during training
        """
        if coordinate is None:
            logger.info("No coordinate system provided. Skipping update.")
            return
        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode != "infevaluate" and mode != "inference":
                    self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            logger.error("An error occurred while updating coordinates: %s", e)
            raise e

    def set_orientation(self, exp_path: Optional[str], data_name: str, rotation_type: Optional[str]) -> None:
        """
        Sets the ground truth orientation based on the provided rotation.
        
        Args:
            exp_path: Path to the pickle file containing orientation data
            data_name: Name of the data sequence
            rotation_type: The type of rotation (AirIMU corrected orientation / raw imu Pre-integration)
        """
        if rotation_type is None or rotation_type == "None" or rotation_type.lower() == "gtrot":
            return
        try:
            with open(exp_path, 'rb') as file:
                loaded_data = pickle.load(file)

            state = loaded_data[data_name]

            if rotation_type.lower() == "airimu":
                self.data["gt_orientation"] = state['airimu_rot']
            elif rotation_type.lower() == "integration":
                self.data["gt_orientation"] = state['inte_rot']
            else:
                logger.error("Unsupported rotation type: %s", rotation_type)
                raise ValueError(f"Unsupported rotation type: {rotation_type}")
        except FileNotFoundError:
            logger.error("The file %s was not found.", exp_path)
            raise

    def remove_gravity(self, remove_g: bool) -> None:
        """
        Remove gravity from accelerometer measurements.
        
        Args:
            remove_g: Whether to remove gravity
        """
        if remove_g is True:
            logger.info("Gravity has been removed")
            self.data["acc"] -= self.g_vector

    @abstractmethod
    def load_imu(self, folder: str) -> None:
        """
        Load IMU data from files.
        
        Args:
            folder: Path to the data folder
        """
        raise NotImplementedError

    @abstractmethod
    def load_gt(self, folder: str) -> None:
        """
        Load ground truth data from files.
        
        Args:
            folder: Path to the data folder
        """
        raise NotImplementedError


class SeqDataset(Data.Dataset):
    """
    A dataset for IMU sequences.
    """
    def __init__(
        self,
        root: str,
        dataname: str,
        devive: str = "cpu",
        name: str = "ALTO",
        duration: Optional[int] = 200,
        step_size: Optional[int] = 200,
        mode: str = "inference",
        drop_last: bool = True,
        conf: Dict[str, Any] = {},
    ):
        """
        Initializes the SeqDataset.

        Args:
            root (str): The root directory of the dataset.
            dataname (str): The name of the data.
            devive (str, optional): The device to use. Defaults to "cpu".
            name (str, optional): The name of the dataset class. Defaults to "ALTO".
            duration (Optional[int], optional): The duration of each sequence segment. Defaults to 200.
            step_size (Optional[int], optional): The step size between segments. Defaults to 200.
            mode (str, optional): The mode of the dataset. Defaults to "inference".
            drop_last (bool, optional): Whether to drop the last incomplete segment. Defaults to True.
            conf (Dict[str, Any], optional): A configuration dictionary. Defaults to {}.
        """
        super().__init__()

        self.DataClass = Sequence.subclasses
        self.DataClass['Euroc'] = self.DataClass.get('EuRoC')
        self.conf = conf
        self.seq = self.DataClass[name](root, dataname, **self.conf)
        self.data = self.seq.data
        self.seqlen = self.seq.__len__() - 1
        self.gravity = conf.gravity if "gravity" in conf.keys() else 9.81007
        self.interpolate = True

        if duration is None:
            self.duration = self.seqlen
        else:
            self.duration = duration

        if step_size is None:
            self.step_size = self.seqlen
        else:
            self.step_size = step_size

        self.data["acc_cov"] = 0.08 * torch.ones_like(self.data["acc"])
        self.data["gyro_cov"] = 0.006 * torch.ones_like(self.data["gyro"])

        start_frame = 0
        end_frame = self.seqlen

        self.index_map = [
            [i, i + self.duration]
            for i in range(0, end_frame - start_frame - self.duration, self.step_size)
        ]
        if (self.index_map[-1][-1] < end_frame) and (not drop_last):
            self.index_map.append([self.index_map[-1][-1], end_frame])

        self.index_map = np.array(self.index_map)

        loaded_param = f"loaded: {root}"
        if "calib" in self.conf:
            loaded_param += f", calib: {self.conf.calib}"
        loaded_param += f", interpolate: {self.interpolate}, gravity: {self.gravity}"
        logger.info(loaded_param)

    def __len__(self) -> int:
        """
        Returns the number of segments in the dataset.
        """
        return len(self.index_map)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        Returns a segment from the dataset.

        Args:
            i (int): The index of the segment.

        Returns:
            Dict[str, torch.Tensor]: The data segment.
        """
        frame_id, end_frame_id = self.index_map[i]
        return {
            "timestamp": self.data['time'][frame_id+1: end_frame_id+1],
            "dt": self.data["dt"][frame_id:end_frame_id],
            "acc": self.data["acc"][frame_id:end_frame_id],
            "gyro": self.data["gyro"][frame_id:end_frame_id],
            "rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "gt_pos": self.data["gt_translation"][frame_id + 1 : end_frame_id + 1],
            "gt_rot": self.data["gt_orientation"][frame_id + 1 : end_frame_id + 1],
            "gt_vel": self.data["velocity"][frame_id + 1 : end_frame_id + 1],
            "init_pos": self.data["gt_translation"][frame_id][None, ...],
            "init_rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "init_vel": self.data["velocity"][frame_id][None, ...],
        }

    def get_init_value(self) -> Dict[str, torch.Tensor]:
        """
        Returns the initial state of the sequence.
        """
        return {
            "pos": self.data["gt_translation"][:1],
            "rot": self.data["gt_orientation"][:1],
            "vel": self.data["velocity"][:1],
        }

    def get_mask(self) -> Any:
        """
        Returns the mask for the data.
        """
        return self.data["mask"]

    def get_gravity(self) -> float:
        """
        Returns the gravity value.
        """
        return self.gravity


class SeqInfDataset(SeqDataset):
    """
    A dataset for inference on IMU sequences, with options to apply corrections.
    """
    def __init__(
        self,
        root: str,
        dataname: str,
        inference_state: Dict[str, Any],
        device: str = "cpu",
        name: str = "ALTO",
        duration: Optional[int] = 200,
        step_size: Optional[int] = 200,
        drop_last: bool = True,
        mode: str = "inference",
        usecov: bool = True,
        useraw: bool = False,
        usetimecut: bool = False,
        conf: Dict[str, Any] = {},
    ):
        """
        Initializes the SeqInfDataset.

        Args:
            root (str): The root directory of the dataset.
            dataname (str): The name of the data.
            inference_state (Dict[str, Any]): The inference state from a model.
            device (str, optional): The device to use. Defaults to "cpu".
            name (str, optional): The name of the dataset class. Defaults to "ALTO".
            duration (Optional[int], optional): The duration of each sequence segment. Defaults to 200.
            step_size (Optional[int], optional): The step size between segments. Defaults to 200.
            drop_last (bool, optional): Whether to drop the last incomplete segment. Defaults to True.
            mode (str, optional): The mode of the dataset. Defaults to "inference".
            usecov (bool, optional): Whether to use covariance from inference state. Defaults to True.
            useraw (bool, optional): Whether to use raw data without corrections. Defaults to False.
            usetimecut (bool, optional): Whether to apply a time cut. Defaults to False.
            conf (Dict[str, Any], optional): A configuration dictionary. Defaults to {}.
        """
        super().__init__(
            root, dataname, device, name, duration, step_size, mode, drop_last, conf
        )
        time_cut = 0
        if usetimecut:
            time_cut = self.seq.time_cut
        if "correction_acc" in inference_state.keys() and not useraw:
            self.data["acc"][:-1] += inference_state["correction_acc"][:, time_cut:].cpu()[0]
            self.data["gyro"][:-1] += inference_state["correction_gyro"][:, time_cut:].cpu()[0]

        if "gyro_bias" in inference_state.keys():
            logger.info(f"Adapted gyro bias: {inference_state['gyro_bias'][time_cut:].cpu()}")
            self.data["gyro"][:-1] -= inference_state["gyro_bias"][time_cut:].cpu()
        if "acc_bias" in inference_state.keys():
            logger.info(f"Adapted acc bias: {inference_state['acc_bias'][time_cut:].cpu()}")
            self.data["acc"][:-1] -= inference_state["acc_bias"][time_cut:].cpu()

        if "adapt_acc" in inference_state.keys():
            self.data["acc"][:-1] -= np.array(inference_state["adapt_acc"][time_cut:])
            self.data["gyro"][:-1] -= np.array(inference_state["adapt_gyro"][time_cut:])

        if "acc_cov" in inference_state.keys() and usecov:
            self.data["acc_cov"] = inference_state["acc_cov"][0][time_cut:]

        if "gyro_cov" in inference_state.keys() and usecov:
            self.data["gyro_cov"] = inference_state["gyro_cov"][0][time_cut:]

    def get_bias(self) -> Dict[str, Any]:
        """
        Returns the gyroscope and accelerometer biases.
        """
        return {"gyro_bias": self.data["g_b"][:-1], "acc_bias": self.data["a_b"][:-1]}
    
    
    def __getitem__(self, i: int) -> Dict[str, Optional[torch.Tensor]]:
        """
        Returns a segment from the dataset.

        Args:
            i (int): The index of the segment.

        Returns:
            Dict[str, Optional[torch.Tensor]]: The data segment.
        """
        frame_id, end_frame_id = self.index_map[i]
        return {
            "acc_cov": self.data["acc_cov"][frame_id:end_frame_id] if "acc_cov" in self.data.keys() else None,
            "gyro_cov": self.data["gyro_cov"][frame_id:end_frame_id] if "gyro_cov" in self.data.keys() else None,
            "timestamp": self.data['time'][frame_id+1: end_frame_id+1],
            "dt": self.data["dt"][frame_id:end_frame_id],
            "acc": self.data["acc"][frame_id:end_frame_id],
            "gyro": self.data["gyro"][frame_id:end_frame_id],
            "rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "gt_pos": self.data["gt_translation"][frame_id + 1 : end_frame_id + 1],
            "gt_rot": self.data["gt_orientation"][frame_id + 1 : end_frame_id + 1],
            "gt_vel": self.data["velocity"][frame_id + 1 : end_frame_id + 1],
            "init_pos": self.data["gt_translation"][frame_id][None, ...],
            "init_rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "init_vel": self.data["velocity"][frame_id][None, ...],
        }


class SeqeuncesDataset(Data.Dataset):
    """
    A dataset for training and inference on multiple IMU sequences.
    
    Features:
    - Abandons the features of the last time frame, since there are no ground truth pose and dt
      to integrate the imu data of the last frame. So the length of the dataset is seq.get_length() - 1.
    """
    @staticmethod
    def find_files(root_dir: str, ext: str = ".csv") -> List[str]:
        """
        Find all files with a specific extension in a directory recursively.
        
        Args:
            root_dir (str): Root directory to search in
            ext (str, optional): File extension to search for (e.g. ".csv", ".parquet"). Defaults to ".csv".
            
        Returns:
            List[str]: List of file paths relative to root_dir
        """
        if not ext.startswith("."):
            ext = f".{ext}"
            
        files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(ext.lower()):  # Case-insensitive extension matching
                    # Get path relative to root_dir
                    rel_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
                    files.append(rel_path)
        return sorted(files)  # Sort for deterministic ordering

    def __init__(
        self,
        data_set_config: Any,
        mode: Optional[str] = None,
        data_path: Optional[str] = None,
        data_root: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        Initializes the SeqeuncesDataset.

        Args:
            data_set_config (Any): The configuration for the dataset.
            mode (Optional[str], optional): The mode of the dataset. Defaults to None.
            data_path (Optional[str], optional): The path to a specific data sequence. Defaults to None.
            data_root (Optional[str], optional): The root directory for multiple sequences. Defaults to None.
            device (str, optional): The device to use. Defaults to "cuda:0".
        """
        super(SeqeuncesDataset, self).__init__()
        (
            self.ts,
            self.dt,
            self.acc,
            self.gyro,
            self.gt_pos,
            self.gt_ori,
            self.gt_velo,
            self.index_map,
            self.seq_idx,
        ) = ([], [], [], [], [], [], [], [], 0)
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        self.device = device
        self.interpolate = True
        self.conf = data_set_config
        self.gravity = self.conf.gravity if "gravity" in self.conf.keys() else 9.81007
        self.weights: List[float] = []

        if mode is None:
            self.mode = data_set_config.mode
        else:
            self.mode = mode

        self.DataClass = Sequence.subclasses
        self.DataClass['Euroc'] = self.DataClass.get('EuRoC')

        ## the design of datapath provide a quick way to revisit a specific sequence, but introduce some inconsistency
        if data_path is None:
            for conf in data_set_config.data_list:
                if "data_drive" in conf:
                    # Use specified data_drive paths
                    for path in conf.data_drive:
                        self.construct_index_map(
                            conf, conf["data_root"], path, self.seq_idx
                        )
                        self.seq_idx += 1
                else:
                    # Auto-discover files in data_root
                    ext = conf.get("file_ext", ".csv")  # Allow configurable extension
                    logger.info(f"No data_drive specified, scanning {conf['data_root']} for *{ext} files")
                    files = self.find_files(conf["data_root"], ext)
                    if not files:
                        logger.warning(f"No *{ext} files found in {conf['data_root']}")
                        continue
                    logger.info(f"Found {len(files)} {ext} files")
                    for file in files:
                        self.construct_index_map(
                            conf, conf["data_root"], file, self.seq_idx
                        )
                        self.seq_idx += 1
        ## the design of dataroot provide a quick way to introduce multiple sequences in eval set, but introduce some inconsistency
        elif data_root is None:
            conf = data_set_config.data_list[0]
            if "data_drive" in conf:
                for data_drive in conf.data_drive:
                    self.construct_index_map(conf, conf["data_root"], data_drive, self.seq_idx)
                    self.seq_idx += 1
            else:
                # Auto-discover files if data_drive not specified
                ext = conf.get("file_ext", ".csv")  # Allow configurable extension
                logger.info(f"No data_drive specified, scanning {conf['data_root']} for *{ext} files")
                files = self.find_files(conf["data_root"], ext)
                if not files:
                    # Fall back to using data_path if provided
                    if data_path:
                        self.construct_index_map(conf, conf["data_root"], data_path, self.seq_idx)
                        self.seq_idx += 1
                    else:
                        logger.warning(f"No *{ext} files found in {conf['data_root']} and no data_path provided")
                else:
                    logger.info(f"Found {len(files)} {ext} files")
                    for file in files:
                        self.construct_index_map(conf, conf["data_root"], file, self.seq_idx)
                        self.seq_idx += 1
        else:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, data_root, data_path, self.seq_idx)
            self.seq_idx += 1


    def load_data(self, seq: Sequence, start_frame: int, end_frame: int) -> None:
        """
        Loads data from a sequence object.

        Args:
            seq (Sequence): The sequence object to load data from.
            start_frame (int): The starting frame index.
            end_frame (int): The ending frame index.
        """
        if "time" in seq.data.keys():
            self.ts.append(seq.data["time"][start_frame:end_frame + 1])
        self.acc.append(seq.data["acc"][start_frame:end_frame])
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])
        # the groud truth state should include the init state and integrated state, thus has one
        # frame than imu data
        self.dt.append(seq.data["dt"][start_frame : end_frame + 1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame : end_frame + 1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame : end_frame + 1])
        self.gt_velo.append(seq.data["velocity"][start_frame : end_frame + 1])

    def construct_index_map(self, conf: Any, data_root: str, data_name: str, seq_id: int) -> None:
        """
        Constructs the index map for a sequence.

        Args:
            conf (Any): The configuration for the sequence.
            data_root (str): The root directory of the data.
            data_name (str): The name of the data sequence.
            seq_id (int): The index of the sequence.
        """
        seq = self.DataClass[conf.name](
            data_root, data_name, interpolate=self.interpolate, **self.conf
        )
        seq_len = len(seq) - 1  # abandon the last imu features
        self.weights.append(conf.weight if 'weight' in conf else 1.)

        start_frame = 0
        end_frame = seq_len
        window_size = self.conf.window_size
        step_size = self.conf.step_size

        if self.mode == "train":
            # For training, we sample random windows
            num_samples = (seq_len // step_size)
            for _ in range(num_samples):
                start = np.random.randint(0, seq_len - window_size)
                end = start + window_size
                self.index_map.append([start, end, seq_id])
                self.load_data(seq, start, end)

        elif self.mode in ["test", "eval", "inference", "infevaluate"]:
            # For testing, evaluation, and inference, we use sliding windows
            index_map = [
                [i, i + window_size, seq_id]
                for i in range(
                    start_frame, end_frame - window_size, step_size
                )
            ]

            if len(index_map) == 0:
                index_map.append([start_frame, end_frame, seq_id])
            elif (index_map[-1][1] < end_frame):
                index_map.append([end_frame - window_size, end_frame, seq_id])
            
            for start, end, seq_id in index_map:
                self.load_data(seq, start, end)
            
            self.index_map.extend(index_map)


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.index_map)

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], int, dict[str, torch.Tensor]]:
        """
        Returns a sample from the dataset.

        Args:
            item (int): The index of the sample.

        Returns:
            tuple[dict[str, torch.Tensor], int, dict[str, torch.Tensor]]: 
                A tuple containing the data, sequence index, and ground truth labels.
        """
        seq_idx = self.index_map[item][2]
        data = {
            "acc": self.acc[item],
            "gyro": self.gyro[item],
            "ts":self.ts[item],
            "dt":self.dt[item],
        }
        label = {
            "gt_pos": self.gt_pos[item],
            "gt_rot": self.gt_ori[item],
            "gt_vel": self.gt_velo[item],
        }
        return data, seq_idx, label

    def get_dtype(self) -> torch.dtype:
        """
        Returns the data type of the tensors.
        """
        return self.acc[0].dtype

    def get_gravity(self) -> float:
        """
        Returns the gravity value.
        """
        return self.gravity


if __name__ == "__main__":
    from datasets.dataset_utils import custom_collate

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.config)
    dataset = SeqeuncesDataset(data_set_config=conf.dataset.train)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=2,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
    )
    for data, init, label in loader:
        print(data["acc"].shape)
        print(init["init_pos"].shape)
        print(label["gt_pos"].shape)
        break
