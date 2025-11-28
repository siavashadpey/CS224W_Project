import torch
from torch import nn

# Instantiate the standard PyTorch MSE loss module
mse_loss_module = nn.MSELoss(reduction='mean')

def l2_loss(predicted_pos: torch.Tensor, ground_truth_pos: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L2 Loss (Mean Squared Error) between predicted 3D coordinates 
    and ground truth 3D coordinates using PyTorch's highly optimized nn.MSELoss.
    
    The loss is calculated as the L2 norm squared over all masked nodes.
    
    Args:
        predicted_pos: Tensor of shape (N, 3) where N is the number of masked nodes.
        ground_truth_pos: Tensor of shape (N, 3).
        
    Returns:
        The scalar MSE loss.
    """
    # nn.MSELoss(reduction='mean') computes: mean( (predicted_pos - ground_truth_pos)^2 )
    # This is exactly the Mean Squared Error over all elements (N * 3).
    # Since coordinates are 3D, this is equivalent to the mean of the squared 
    # Euclidean distance (L2 norm squared) for each point.
    return mse_loss_module(predicted_pos, ground_truth_pos)