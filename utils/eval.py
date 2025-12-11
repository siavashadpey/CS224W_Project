import torch 
import torch.nn.functional as F

def pearson_correlation_coefficient(y_pred: torch.Tensor,
                                    y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson Correlation Coefficient between true and predicted values.

    Args:
        y_pred (Tensor): Predicted values.
        y_true (Tensor): Ground truth values.
    Returns:
        Tensor: Pearson Correlation Coefficient.
    """
    y_pred_mean = torch.mean(y_pred)
    y_true_mean = torch.mean(y_true)
    y_pred_centered = y_pred - y_pred_mean
    y_true_centered = y_true - y_true_mean
    covariance = torch.sum(y_pred_centered * y_true_centered)
    y_pred_std = torch.sqrt(torch.sum(y_pred_centered ** 2))
    y_true_std = torch.sqrt(torch.sum(y_true_centered ** 2))
    denominator = y_pred_std * y_true_std + 1e-8
    pearson_correlation = covariance / denominator
    return pearson_correlation

def scaled_pK_rmse(y_pred: torch.Tensor,
                   y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the RMSE between true and predicted pK (-log affinity) values.
    Args:
        y_pred (Tensor): Predicted pK values.
        y_true (Tensor): Ground truth pK values.
    """
    # Convention is to scale from 0 to 16.
    min = 0
    max = 16
    y_pred_scaled = y_pred * (max - min) + min
    y_true_scaled = y_true * (max - min) + min
    return torch.sqrt(F.mse_loss(y_pred_scaled, y_true_scaled))
  