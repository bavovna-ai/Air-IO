import torch
from typing import Callable, Dict

EPSILON = 1e-7

def diag_cov_loss(dist: torch.Tensor, pred_cov: torch.Tensor) -> torch.Tensor:
    """
    Calculates the diagonal covariance loss.

    Args:
        dist (torch.Tensor): The distance tensor.
        pred_cov (torch.Tensor): The predicted covariance tensor.

    Returns:
        torch.Tensor: The calculated loss.
    """
    error = (dist).pow(2)
    return torch.mean(error / 2*(torch.exp(2 * pred_cov)) + pred_cov)

def diag_ln_cov_loss(dist: torch.Tensor, pred_cov: torch.Tensor, use_epsilon: bool = False) -> torch.Tensor:
    """
    Calculates the diagonal log covariance loss.

    Args:
        dist (torch.Tensor): The distance tensor.
        pred_cov (torch.Tensor): The predicted covariance tensor.
        use_epsilon (bool, optional): Whether to use an epsilon for numerical stability. 
                                      Defaults to False.

    Returns:
        torch.Tensor: The calculated loss.
    """
    error = (dist).pow(2)
    if use_epsilon: l = ((error / pred_cov) + torch.log(pred_cov + EPSILON))
    else: l = ((error / pred_cov) + torch.log(pred_cov))
    return l.mean()

def L2(dist: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L2 loss (mean squared error).

    Args:
        dist (torch.Tensor): The distance tensor.

    Returns:
        torch.Tensor: The L2 loss.
    """
    error = dist.pow(2)
    return torch.mean(error)

def L1(dist: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L1 loss (mean absolute error).

    Args:
        dist (torch.Tensor): The distance tensor.

    Returns:
        torch.Tensor: The L1 loss.
    """
    error = (dist).abs().mean()
    return error

def loss_weight_decay(error: torch.Tensor, decay_rate: float = 0.95) -> torch.Tensor:
    """
    Applies weight decay to the loss.

    Args:
        error (torch.Tensor): The error tensor.
        decay_rate (float, optional): The decay rate. Defaults to 0.95.

    Returns:
        torch.Tensor: The error with weight decay applied.
    """
    F = error.shape[-2]
    decay_list = decay_rate * torch.ones(F, device=error.device, dtype=error.dtype)
    decay_list[0] = 1.
    decay_list = torch.cumprod(decay_list, 0)
    error = torch.einsum('bfc, f -> bfc', error, decay_list)
    return error

def loss_weight_decrease(error: torch.Tensor, decay_rate: float = 0.95) -> torch.Tensor:
    """
    Applies decreasing weight to the loss.

    Args:
        error (torch.Tensor): The error tensor.
        decay_rate (float, optional): The decay rate (unused). Defaults to 0.95.

    Returns:
        torch.Tensor: The error with decreasing weight applied.
    """
    F = error.shape[-2]
    decay_list = torch.tensor([1./i for i in range(1, F+1)], device=error.device, dtype=error.dtype)
    error = torch.einsum('bfc, f -> bfc', error, decay_list)
    return error

def Huber(dist: torch.Tensor, delta: float = 0.005) -> torch.Tensor:
    """
    Calculates the Huber loss.

    Args:
        dist (torch.Tensor): The distance tensor.
        delta (float, optional): The delta for the Huber loss. Defaults to 0.005.

    Returns:
        torch.Tensor: The Huber loss.
    """
    error = torch.nn.functional.huber_loss(dist, torch.zeros_like(dist, device=dist.device), delta=delta)
    return error


loss_fc_list: Dict[str, Callable[..., torch.Tensor]] = {
    "L2": L2,
    "L1": L1,
    "diag_cov_ln": diag_ln_cov_loss,
    "Huber_loss005":lambda dist: Huber(dist, delta = 0.005),
    "Huber_loss05":lambda dist: Huber(dist, delta = 0.05),
}