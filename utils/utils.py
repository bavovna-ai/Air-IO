import os
import torch
import io, pickle
import numpy as np
from inspect import currentframe, getframeinfo
import wandb
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import Dict, Any, List, Union
from numpy.typing import NDArray
import warnings

def save_state(out_states: Dict[str, List[Any]], in_state: Dict[str, Any]) -> None:
    """
    Recursively saves the values from in_state to out_states.

    Args:
        out_states (Dict[str, List[Any]]): The dictionary to save states to.
        in_state (Dict[str, Any]): The dictionary of states to save.
    """
    for k, v in in_state.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            save_state(out_states=out_states, in_state=v)
        elif k in out_states.keys():
            out_states[k].append(v)
        else:
            out_states[k] = [v]

def Gaussian_noise(num_nodes: int, 
                   sigma_x: float = 0.05, 
                   sigma_y: float = 0.05, 
                   sigma_z: float = 0.05) -> torch.Tensor:
    """
    Generates 3D Gaussian noise.

    Args:
        num_nodes (int): The number of noise samples to generate.
        sigma_x (float, optional): The standard deviation in the x-axis. Defaults to 0.05.
        sigma_y (float, optional): The standard deviation in the y-axis. Defaults to 0.05.
        sigma_z (float, optional): The standard deviation in the z-axis. Defaults to 0.05.

    Returns:
        torch.Tensor: A tensor of shape (num_nodes, 3) with Gaussian noise.
    """
    std = torch.stack([torch.ones(num_nodes)*sigma_x, torch.ones(num_nodes)*sigma_y, torch.ones(num_nodes)*sigma_z], dim=-1)
    return torch.normal(mean = 0, std = std)

def move_to(obj: Any, device: torch.device) -> Any:
    """
    Moves a tensor, list, or dictionary of tensors to the specified device.

    Args:
        obj (Any): The object to move. Can be a tensor, list, or dictionary.
        device (torch.device): The target device.

    Returns:
        Any: The object moved to the specified device.
    """
    if torch.is_tensor(obj):return obj.to(device)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj).to(device)
    else:
        raise TypeError("Invalid type for move_to", type(obj))

def qinterp(qs: torch.Tensor, t: NDArray[np.float64], t_int: NDArray[np.float64]) -> torch.Tensor:
    """
    Interpolates quaternions using spherical linear interpolation (Slerp).

    Args:
        qs (torch.Tensor): A tensor of quaternions to interpolate.
        t (NDArray[np.float64]): The timestamps for the given quaternions.
        t_int (NDArray[np.float64]): The timestamps to interpolate to.

    Returns:
        torch.Tensor: The interpolated quaternions.
    """
    qs_np = qs.numpy()
    slerp = Slerp(t, R.from_quat(qs_np))
    interp_rot = slerp(t_int).as_quat()
    return torch.tensor(interp_rot)

def interp_xyz(time: NDArray[np.float64], 
               opt_time: NDArray[np.float64], 
               xyz: NDArray[np.float64]) -> torch.Tensor:
    """
    Interpolates 3D coordinates.

    Args:
        time (NDArray[np.float64]): The timestamps to interpolate to.
        opt_time (NDArray[np.float64]): The timestamps of the original coordinates.
        xyz (NDArray[np.float64]): The 3D coordinates to interpolate.

    Returns:
        torch.Tensor: The interpolated 3D coordinates as a tensor.
    """
    intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
    intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
    intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])
    inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
    return torch.tensor(inte_xyz)

def lookAt(dir_vec: Union[torch.Tensor, NDArray[np.float64]], 
           up: torch.Tensor = torch.tensor([0.,0.,1.], dtype=torch.float64), 
           source: torch.Tensor = torch.tensor([0.,0.,0.], dtype=torch.float64)) -> torch.Tensor:
    """
    Computes a rotation matrix that looks from a source point towards a direction vector.

    Args:
        dir_vec (Union[torch.Tensor, NDArray[np.float64]]): The direction vector.
        up (torch.Tensor, optional): The up vector. Defaults to torch.tensor([0.,0.,1.]).
        source (torch.Tensor, optional): The source point. Defaults to torch.tensor([0.,0.,0.]).

    Returns:
        torch.Tensor: The resulting rotation matrix.
    """
    if not isinstance(dir_vec, torch.Tensor):
        dir_vec = torch.tensor(dir_vec)
    def normalize(x: torch.Tensor) -> torch.Tensor:
        length = x.norm()
        if length< 0.005:
            length = 1
            warnings.warn('Normlization error that the norm is too small')
        return x/length
            
    zaxis = normalize(dir_vec - source)
    xaxis = normalize(torch.cross(zaxis, up))
    yaxis = torch.cross(xaxis, zaxis)

    m = torch.tensor([
        [xaxis[0], xaxis[1], xaxis[2]],
        [yaxis[0], yaxis[1], yaxis[2]],
        [zaxis[0], zaxis[1], zaxis[2]],
    ])

    return m

def cat_state(in_state: Dict[str, List[torch.Tensor]]) -> None:
    """
    Concatenates lists of tensors in a state dictionary.

    Args:
        in_state (Dict[str, List[torch.Tensor]]): The dictionary of states to concatenate.
    """
    pop_list = []
    for k, v in in_state.items():
        if v and len(v[0].shape) > 2:
            in_state[k] = torch.cat(v,  dim=-2)
        else:
            pop_list.append(k)
    for k in pop_list:
        in_state.pop(k)

class CPU_Unpickler(pickle.Unpickler):
    """
    A custom unpickler that maps all tensors to the CPU.
    """
    def find_class(self, module: str, name: str) -> Any:
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def write_board(writer: Any, objs: Union[Dict[str, float], float], epoch_i: int, header: str = '') -> None:
    """
    Writes data to a tensorboard writer.

    Args:
        writer: The tensorboard writer object.
        objs (Union[Dict[str, float], float]): The object to write.
        epoch_i (int): The current epoch.
        header (str, optional): A header string for the log. Defaults to ''.
    """
    # writer = SummaryWriter(log_dir=conf.general.exp_dir)
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                writer.add_scalar(os.path.join(header, k), v, epoch_i)
    elif isinstance(objs, float):
        writer.add_scalar(header, v, epoch_i)

def write_wandb(header: str, objs: Any, epoch_i: int) -> None:
    """
    Logs data to Weights & Biases.

    Args:
        header (str): The header for the log.
        objs (Any): The object to log.
        epoch_i (int): The current step or epoch.
    """
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                wandb.log({os.path.join(header, k): v}, epoch_i)
    else:
        wandb.log({header: objs}, step = epoch_i)

def save_ckpt(network: torch.nn.Module, 
              optimizer: torch.optim.Optimizer, 
              scheduler: Any, 
              epoch_i: int, 
              test_loss: float, 
              conf: Any, 
              save_best: bool = False) -> None:
    """
    Saves a model checkpoint.

    Args:
        network (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        scheduler (Any): The scheduler state to save.
        epoch_i (int): The current epoch.
        test_loss (float): The test loss.
        conf (Any): The configuration object.
        save_best (bool, optional): Whether to save this as the best model. Defaults to False.
    """
    if epoch_i % conf.train.save_freq == conf.train.save_freq-1:
        torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/%04d.ckpt"%epoch_i))

    if save_best:
        print("saving the best model", test_loss)
        torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt"))
    
    torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))


def report_hasNan(x: torch.Tensor) -> None:
    """
    Reports if a tensor contains NaN values.

    Args:
        x (torch.Tensor): The tensor to check.
    """
    cf = currentframe().f_back
    if cf is None:
        return
    res = torch.any(torch.isnan(x)).cpu().item()
    if res: print(f"[hasnan!] {getframeinfo(cf).filename}:{cf.f_lineno}")


def report_hasNeg(x: torch.Tensor) -> None:
    """
    Reports if a tensor contains negative values.

    Args:
        x (torch.Tensor): The tensor to check.
    """
    cf = currentframe().f_back
    if cf is None:
        return
    res = torch.any(x < 0).cpu().item()
    if res: print(f"[hasneg!] {getframeinfo(cf).filename}:{cf.f_lineno}")


    