import argparse
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import itertools

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory
from .dataset import Sequence, SeqeuncesDataset
import pypose as pp

# Configure logging
logger = logging.getLogger(__name__)

class SeqeuncesMotionDataset(SeqeuncesDataset):
    def __init__(
        self,
        data_set_config: ConfigFactory,
        mode: Optional[str] = None,
        data_path: Optional[str] = None,
        data_root: Optional[str] = None,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(
            data_set_config=data_set_config,
            mode=mode,
            data_path=data_path,
            data_root=data_root,
            device=device,
        )
        logger.info(f"Loading {data_set_config.mode} dataset")
        logger.info(f"Loaded: {data_set_config.data_list[0]['data_root']}")
        
        if "coordinate" in data_set_config:
            logger.info(f"Coordinate: {data_set_config.coordinate}")
        if "remove_g" in data_set_config and data_set_config.remove_g is True:
            logger.info("Gravity has been removed")
        if "rot_type" in data_set_config:
            if data_set_config.rot_type is None:
                logger.info("Using groundtruth orientation")
            elif data_set_config.rot_type.lower() == "airimu":
                logger.info(f"Using AirIMU orientation loaded from {data_set_config.rot_path}")
            elif data_set_config.rot_type.lower() == "integration":
                logger.info(f"Using pre-integration orientation loaded from {data_set_config.rot_path}")
        logger.info(f"Gravity: {data_set_config.gravity}")

        # Get feature configuration from model config if available
        self.feature_config = getattr(data_set_config, 'features', None)
        if self.feature_config is None and hasattr(data_set_config, 'model'):
            self.feature_config = data_set_config.model.features
        
        # Calculate feature dimensions and names once during initialization
        self._feature_names = self._get_feature_names()
        self._n_features = self._calculate_feature_dim()
        
        logger.info(f"Feature names: {self._feature_names}")
        logger.info(f"Number of features: {self._n_features}")

    def _calculate_feature_dim(self) -> int:
        """
        Calculate total number of input features based on loaded data.
        
        Returns:
            int: Total number of input features
        """
        if not self._feature_names:
            raise ValueError("Feature names must be computed before calculating dimensions")
            
        feature_dim = 0
        for feature_name in self._feature_names:
            data = getattr(self, feature_name)
            if data and len(data) > 0:
                if len(data[0].shape) > 1:
                    feature_dim += data[0].shape[-1]
                else:
                    feature_dim += 1
                    
        return feature_dim

    def _get_feature_names(self) -> List[str]:
        """
        Get list of feature names in the order they should be concatenated.
        
        Returns:
            List[str]: List of feature names
        """
        if not self.feature_config:
            # Default to basic IMU features if no config provided
            return ["acc", "gyro"]
            
        feature_names = []
        
        # Add IMU features first (required)
        if "imu" in self.feature_config:
            if "acc" in self.feature_config.imu:
                feature_names.append("acc")
            if "gyro" in self.feature_config.imu:
                feature_names.append("gyro")
                
        # Add additional features in config order
        if "additional" in self.feature_config:
            for feature_name in self.feature_config.additional:
                if hasattr(self, feature_name):
                    data = getattr(self, feature_name)
                    if data and len(data) > 0:
                        feature_names.append(feature_name)
                    
        return feature_names

    @property
    def n_features(self) -> int:
        """Total number of input features."""
        return self._n_features

    @property
    def feature_names(self) -> List[str]:
        """List of feature names in concatenation order."""
        return self._feature_names.copy()  # Return a copy to prevent modification

    def validate_features(self, required_features: List[str]) -> None:
        """
        Validate that all required features are available in the dataset.
        
        Args:
            required_features: List of feature names required by the model
            
        Raises:
            ValueError: If any required features are missing
        """
        missing = []
        for feature in required_features:
            if feature not in self._feature_names:
                missing.append(feature)
            elif not hasattr(self, feature) or not getattr(self, feature):
                missing.append(feature)
                
        if missing:
            raise ValueError(f"Dataset is missing required features: {missing}")

    def load_data(self, seq: Sequence, start_frame: int, end_frame: int) -> None:
        self.ts.append(seq.data["time"][start_frame:end_frame + 1])
        self.acc.append(seq.data["acc"][start_frame:end_frame])
        self.gyro.append(seq.data["gyro"][start_frame:end_frame])
        self.dt.append(seq.data["dt"][start_frame : end_frame + 1])
        self.gt_pos.append(seq.data["gt_translation"][start_frame : end_frame + 1])
        self.gt_ori.append(seq.data["gt_orientation"][start_frame : end_frame + 1])
        self.gt_velo.append(seq.data["velocity"][start_frame : end_frame + 1])

    def construct_index_map(self, conf: ConfigFactory, data_root: str, data_name: str, seq_id: int) -> None:
        seq = self.DataClass[conf.name](
            data_root, data_name, **self.conf
        )
        seq_len = len(seq) - 1
        window_size, step_size = conf.window_size, conf.step_size
        start_frame, end_frame = 0, seq_len

        if self.mode == 'train_70':
            end_frame = np.floor(seq_len * 0.7).astype(int)
        elif self.mode == 'test_30':
            start_frame = np.floor(seq_len * 0.7).astype(int)

        _duration = end_frame - start_frame
        if self.mode == "inference":
            window_size = seq_len
            step_size = seq_len
            self.index_map = [[seq_id, 0, seq_len]]
        elif self.mode == "infevaluate":
            self.index_map += [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size, step_size)
            ]
            if self.index_map[-1][2] < _duration:
                self.index_map += [[seq_id, self.index_map[-1][2], seq_len]]
        elif self.mode == "evaluate":
            self.index_map += [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size, step_size)
            ]
        else:
            sub_index_map = [
                [seq_id, j, j + window_size]
                for j in range(0, _duration - window_size - step_size, step_size)
                if torch.all(seq.data["mask"][j : j + window_size])
            ]
            self.index_map += sub_index_map

        self.load_data(seq, start_frame, end_frame)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        seq_id, frame_id, end_frame_id = self.index_map[item]
        data = {
            'timestamp': self.ts[seq_id][frame_id:end_frame_id + 1],
            'dt': self.dt[seq_id][frame_id: end_frame_id+1],
            'acc': self.acc[seq_id][frame_id: end_frame_id],
            'gyro': self.gyro[seq_id][frame_id: end_frame_id],
            'rot': self.gt_ori[seq_id][frame_id: end_frame_id]
        }
        init_state = {
            'init_rot': self.gt_ori[seq_id][frame_id][None, ...],
            'init_pos': self.gt_pos[seq_id][frame_id][None, ...],
            'init_vel': self.gt_velo[seq_id][frame_id][None, ...],
        }
        label = {
            'gt_pos': self.gt_pos[seq_id][frame_id : end_frame_id+1],
            'gt_rot': self.gt_ori[seq_id][frame_id : end_frame_id+1],
            'gt_vel': self.gt_velo[seq_id][frame_id : end_frame_id+1],
        }
        return {**data, **init_state, **label}

    def get_init_value(self) -> Dict[str, torch.Tensor]:
        return {
            "pos": self.data["gt_translation"][:1],
            "rot": self.data["gt_orientation"][:1],
            "vel": self.data["velocity"][:1],
        }

if __name__ == "__main__":
    from datasets.dataset_utils import custom_collate

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/BaselineEuRoC.conf",
        help="config file path, i.e., configs/Euroc.conf",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")

    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.config)

    dataset = SeqeuncesMotionDataset(data_set_config=conf.train)
    loader = Data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate
    )

    for i, (data, init, _label) in enumerate(loader):
        for k in data:
            logger.info(f"{k}: {data[k].shape}")
        for k in init:
            logger.info(f"{k}: {init[k].shape}")
