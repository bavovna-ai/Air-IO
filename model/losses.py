import torch
from .loss_func import loss_fc_list, diag_ln_cov_loss
from utils import report_hasNan
import numpy as np
from typing import Callable, Dict, Any, Tuple

def motion_loss_(fc: Callable[..., torch.Tensor], 
                 pred: torch.Tensor, 
                 targ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the motion loss and distance.

    Args:
        fc (Callable[..., torch.Tensor]): The loss function.
        pred (torch.Tensor): The predicted tensor.
        targ (torch.Tensor): The target tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The calculated loss and the distance.
    """
    dist = pred - targ
    loss = fc(dist)
    return loss, dist

def get_motion_loss(inte_state: Dict[str, torch.Tensor], 
                    label: torch.Tensor, 
                    confs: Any) -> Dict[str, torch.Tensor]:
    """
    Calculates the total motion loss.

    Args:
        inte_state (Dict[str, torch.Tensor]): The integrated state from the model.
        label (torch.Tensor): The ground truth label.
        confs (Any): The configuration object.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the total loss and covariance loss.
    """
    ## The state loss for evaluation
    loss = torch.tensor(0.0, device=label.device)
    cov_loss = torch.tensor(0.0, device=label.device)
    loss_fc = loss_fc_list[confs.loss]
    
    vel_loss, vel_dist = motion_loss_(loss_fc, inte_state['net_vel'],label)

    # Apply the covariance loss
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov)
    loss += confs.weight * vel_loss
    return {'loss':loss, 'cov_loss':cov_loss}


def get_motion_RMSE(inte_state: Dict[str, torch.Tensor], 
                    label: torch.Tensor, 
                    confs: Any) -> Dict[str, torch.Tensor]:
    """
    Calculates the Root Mean Square Error (RMSE) for motion.

    Args:
        inte_state (Dict[str, torch.Tensor]): The integrated state from the model.
        label (torch.Tensor): The ground truth label.
        confs (Any): The configuration object.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the RMSE loss, distance, covariance loss,
                                and Pearson correlation coefficients for each velocity component.
    """
    def _RMSE(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((x.norm(dim=-1)**2).mean())
        
    def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate Pearson R for each velocity component."""
        # x, y shape: [batch, time, 3]
        vx = x - x.mean(dim=1, keepdim=True)
        vy = y - y.mean(dim=1, keepdim=True)
        r = (vx * vy).sum(dim=1) / (torch.sqrt((vx ** 2).sum(dim=1) * (vy ** 2).sum(dim=1)) + 1e-8)
        return r.mean(dim=0)  # Average across batch, return [3] for x,y,z components
        
    cov_loss = torch.tensor(0.0, device=label.device)
    pred_vel = inte_state['net_vel']
    dist = (pred_vel - label)
    dist = torch.mean(dist, dim=-2)
    loss = _RMSE(dist)[None,...]
    
    # Calculate Pearson R for velocities
    pearson_r = _pearson_r(pred_vel, label)  # [3] tensor for x,y,z components
    
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()
    
    return {
        'loss': loss, 
        'dist': dist.norm(dim=-1).mean(),
        'cov_loss': cov_loss,
        'pearson_r': pearson_r  # [x,y,z] correlations
    }
