"""
Code for handling model configurations and feature extraction.
"""
from typing import Dict, List, Any, Optional, Sequence
import logging
import torch
import torch.nn as nn
from .config_defaults import merge_with_defaults

logger = logging.getLogger(__name__)

def process_config(config: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
    """
    Process configuration by merging with dataset-provided defaults.
    
    Args:
        config: Raw configuration dictionary
        dataset: Optional dataset instance to get feature configuration from
        
    Returns:
        Processed configuration with defaults
    """
    # Get dataset name from train section
    if "train" not in config or "data_list" not in config["train"]:
        raise ValueError("Configuration must have train.data_list section")
    
    # If no model section exists, create it
    if "model" not in config:
        config["model"] = {}
    
    # If features not defined and dataset provided, use dataset's feature configuration
    if "features" not in config["model"] and dataset is not None:
        config["model"]["features"] = dataset.feature_dict
        config["model"]["n_features"] = len(dataset.feature_dict) if dataset.feature_dict is not None else 6
    
    return config

class CNNEncoder(nn.Module):
    """A 1D CNN encoder module."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int] = [16, 32, 64, 128],
        kernel_sizes: Sequence[int] = [7, 7, 7, 7],
        strides: Sequence[int] = [1, 1, 1, 1],
        paddings: Sequence[int] = [3, 3, 3, 3]
    ):
        super().__init__()
        
        # Validate input parameters
        if not hidden_channels:
            raise ValueError("hidden_channels cannot be empty")
            
        n_layers = len(hidden_channels)
        if len(kernel_sizes) != n_layers:
            raise ValueError(f"Expected {n_layers} kernel sizes, got {len(kernel_sizes)}")
        if len(strides) != n_layers:
            raise ValueError(f"Expected {n_layers} stride values, got {len(strides)}")
        if len(paddings) != n_layers:
            raise ValueError(f"Expected {n_layers} padding values, got {len(paddings)}")
            
        channels = [in_channels] + list(hidden_channels)
        layers = []

        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv1d(
                    channels[i],
                    channels[i+1],
                    kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i]
                )
            )
            layers.append(nn.BatchNorm1d(channels[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CodeNetMotion(nn.Module):
    """Basic motion estimation network."""
    def __init__(
        self,
        input_dim: int,
        feature_names: Sequence[str],
        hidden_channels: Sequence[int] = [32, 64],
        propcov: bool = True,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.propcov = propcov
        
        # Keep these as instance variables for get_label method
        self.k_list = [7, 7]
        self.p_list = [3, 3]
        self.s_list = [3, 3]
        
        self.cnn = CNNEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_channels,
            kernel_sizes=self.k_list,
            strides=self.s_list,
            paddings=self.p_list
        )
        
        last_cnn_channels = hidden_channels[-1]
        self.gru1 = nn.GRU(
            input_size=last_cnn_channels,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.gru2 = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.veldecoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        self.velcov_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

    def _validate_input(self, data: Dict[str, torch.Tensor]) -> None:
        """Validate that all required features are present in input data."""
        missing = [f for f in self.feature_names if f not in data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        x = self.cnn(x.transpose(-1, -2)).transpose(-1, -2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the velocity from the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The decoded velocity.
        """
        vel = self.veldecoder(x)
        return vel

    def cov_decoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the covariance from the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The decoded covariance.
        """
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def get_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        """
        Gets the appropriate label for the output.

        Args:
            gt_label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The selected label.
        """
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0] * self.s_list[1], :]
        L_out = (gt_label.shape[1] - 1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label, gt_label[:,-1:,:].repeat(1, diff, 1)), dim=1)
        return select_label

    def forward(self, data: Dict[str, torch.Tensor], rot: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        """
        Performs the forward pass of the model.

        Args:
            data (Dict[str, torch.Tensor]): The input data dictionary.
            rot (Optional[torch.Tensor], optional): The rotation tensor. Defaults to None.

        Returns:
            Dict[str, Optional[torch.Tensor]]: A dictionary containing the network output.
        """
        self._validate_input(data)
        
        # Concatenate features in the order specified
        features = [data[name] for name in self.feature_names]
        feature = torch.cat(features, dim=-1)
        
        feature = self.encoder(feature)
        net_vel = self.decoder(feature)
        
        cov = None
        if self.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 'net_vel': net_vel}

    
class CodeNetMotionwithRot(nn.Module):
    """Motion estimation network with rotation handling.
    
    This network processes both feature data and rotation data through separate encoders,
    then combines them for motion estimation. It supports covariance propagation and
    handles sequence length adjustments due to convolutions.
    """
    def __init__(
        self,
        input_dim: int,
        input_dim_ori: int = 3,
        feature_names: Sequence[str] = None,
        hidden_channels: Sequence[int] = [32, 64],
        kernel_sizes: Sequence[int] = [7, 7],
        strides: Sequence[int] = [3, 3],
        padding_num: int = 3,
        propcov: bool = True
    ):
        """Initialize the CodeNetMotionwithRot model.
        
        Args:
            input_dim (int): Number of input feature dimensions
            input_dim_ori (int, optional): Number of orientation input dimensions.
                Defaults to 3.
            feature_names (Sequence[str], optional): Names of input features.
                Defaults to None.
            hidden_channels (Sequence[int], optional): Number of channels in each conv layer. 
                Defaults to [32, 64].
            kernel_sizes (Sequence[int], optional): Kernel sizes for conv layers. 
                Defaults to [7, 7].
            strides (Sequence[int], optional): Stride values for conv layers. 
                Defaults to [3, 3].
            padding_num (int, optional): Padding size for conv layers. 
                Defaults to 3.
            propcov (bool, optional): Whether to propagate covariance. 
                Defaults to True.
        """
        super().__init__()
        
        # Validate parameters
        if not hidden_channels:
            raise ValueError("hidden_channels cannot be empty")
        if len(kernel_sizes) != len(hidden_channels):
            raise ValueError(f"Expected {len(hidden_channels)} kernel sizes, got {len(kernel_sizes)}")
        if len(strides) != len(hidden_channels):
            raise ValueError(f"Expected {len(hidden_channels)} stride values, got {len(strides)}")
            
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.propcov = propcov
        
        # Network architecture parameters
        self.interval = 9  # Used for sequence processing
        self.padding_num = padding_num
        self.k_list = kernel_sizes
        self.s_list = strides
        self.p_list = [padding_num] * len(kernel_sizes)
        
        # Separate encoders for features and rotation
        self.feature_encoder = CNNEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_channels,
            kernel_sizes=self.k_list,
            strides=self.s_list,
            paddings=self.p_list
        )
        self.ori_encoder = CNNEncoder(
            in_channels=input_dim_ori,
            hidden_channels=hidden_channels,
            kernel_sizes=self.k_list,
            strides=self.s_list,
            paddings=self.p_list
        )
        
        # GRU layers for sequence processing
        self.gru1 = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.gru2 = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Additional processing layers
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128, 64))
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.gelu = nn.GELU()
        
        # Decoder networks
        self.veldecoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        self.velcov_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

    def _validate_input(self, data: Dict[str, torch.Tensor]) -> None:
        """Validate that all required features are present in input data."""
        if self.feature_names:  # Only validate if feature names were provided
            missing = [f for f in self.feature_names if f not in data]
            if missing:
                raise ValueError(f"Missing required features: {missing}")

    def encoder(self, feature: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input feature and orientation tensors.

        Args:
            feature (torch.Tensor): The feature tensor.
            ori (torch.Tensor): The orientation tensor.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1, -2)).transpose(-1, -2)
        x = torch.cat([x1, x2], dim=-1)
        
        x = self.fcn2(x)
        x = self.batchnorm2(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.gelu(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the velocity from the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The decoded velocity.
        """
        vel = self.veldecoder(x)
        return vel

    def cov_decoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the covariance from the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The decoded covariance.
        """
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def get_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        """
        Gets the appropriate label for the output.

        Args:
            gt_label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The selected label.
        """
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0] * self.s_list[1], :]
        L_out = (gt_label.shape[1] - 1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label, gt_label[:,-1:,:].repeat(1, diff, 1)), dim=1)
        return select_label

    def forward(self, data: Dict[str, torch.Tensor], rot: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        """
        Performs the forward pass of the model.

        Args:
            data (Dict[str, torch.Tensor]): The input data dictionary.
            rot (Optional[torch.Tensor], optional): The rotation tensor. Defaults to None.

        Returns:
            Dict[str, Optional[torch.Tensor]]: A dictionary containing the network output.
        """
        self._validate_input(data)
        if rot is None:
            raise ValueError("Rotation tensor is required for this model")
        
        # Concatenate features in the order specified
        if self.feature_names:
            features = [data[name] for name in self.feature_names]
            feature = torch.cat(features, dim=-1)
        else:
            # If no feature names specified, assume data is already concatenated
            feature = next(iter(data.values()))
        
        feature = self.encoder(feature, rot)
        net_vel = self.decoder(feature)
        
        cov = None
        if self.propcov:
            cov = self.cov_decoder(feature)
        
        return {"cov": cov, "net_vel": net_vel}
