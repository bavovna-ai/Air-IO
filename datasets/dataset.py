import argparse
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory


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

    @abstractmethod
    def get_length(self) -> int:
        """
        Returns the length of the sequence.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> Dict[str, Any]:
        """
        Returns the sequence data.
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
        self.conf = conf
        self.seq = self.DataClass[name](root, dataname, **self.conf)
        self.data = self.seq.data
        self.seqlen = self.seq.get_length() - 1
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
        print(loaded_param)

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
            print("adapted gyro bias: ", inference_state["gyro_bias"][time_cut:].cpu())
            self.data["gyro"][:-1] -= inference_state["gyro_bias"][time_cut:].cpu()
        if "acc_bias" in inference_state.keys():
            print("adapted acc bias: ", inference_state["acc_bias"][time_cut:].cpu())
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

        ## the design of datapath provide a quick way to revisit a specific sequence, but introduce some inconsistency
        if data_path is None:
            for conf in data_set_config.data_list:
                for path in conf.data_drive:
                    self.construct_index_map(
                        conf, conf["data_root"], path, self.seq_idx
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
                 self.construct_index_map(conf, conf["data_root"], data_path, self.seq_idx)
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
        seq_len = seq.get_length() - 1  # abandon the last imu features
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
