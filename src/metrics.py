import numpy as np
from tqdm import tqdm
import torch

# Telecommunication imports
from .power_allocation import *

epsilon = 1e-12

class SumRateLoss(torch.nn.Module):
    """
    Criterion class for Sum Rate as Loss.

    It returns the sum rate as loss using the wireless channels and power

    Attributes:
    P_total (float): Total power for the system
    noise_variance (float): Sigma^2 
    """
    def __init__(self, P_total: float=1.0, noise_variance: float=1.0):
        """
        Constructor

        Inputs:
        P_total (float): Total power for the system
        noise_variance (float): Sigma^2 
        """
        super().__init__()
        self.P_total = P_total
        self.noise_variance = noise_variance

    def forward(self, pred_power, H):
        """
        Uses the negative sum rate as loss function
        
        Inputs
        pred_power (torch.tensor): Predicted power by the model [B, S, K]
        H (torch.tensor): Original channel [B, S, K, N]

        Output:
        loss (Scalar): negative sum rate
        """
        # 1. Scale power to comply with power budget
        power = pred_power * self.P_total

        # 2. Construct directions using MRT precoder
        with torch.no_grad():
            V_mrt = get_mrt_directions(H)

        # 3. Combine power and directions
        W_pred = V_mrt * torch.sqrt(power.unsqueeze(-2) + epsilon)

        # 4. Calculate sum rate
        sum_rate_batch = calculate_sum_rate(H, W_pred, self.noise_variance)

        # 5. Invert (We want to minimize)
        loss = -torch.mean(sum_rate_batch)

        return loss
    
class PrecoderSumRateLoss(torch.nn.Module):
    """
    Criterion class for Sum Rate as Loss.

    It returns the sum rate as loss using the wireless channels and precoder

    Attributes:
    P_total (float): Total power for the system
    noise_variance (float): Sigma^2 
    """
    def __init__(self, P_total: float=1.0, noise_variance: float=1.0):
        """
        Constructor

        Inputs:
        P_total (float): Total power for the system
        noise_variance (float): Sigma^2 
        """
        super().__init__()
        self.P_total = P_total
        self.noise_variance = noise_variance

    def forward(self, W_pred, H):
        """
        Uses the negative sum rate as loss function
        
        Inputs
        W (torch.tensor): Predicted precoder [B, S, N, K]
        H (torch.tensor): Original channel [B, S, K, N]

        Output:
        loss (Scalar): negative sum rate
        """

        # 1. Calculate sum rate
        sum_rate_batch = calculate_sum_rate(H, W_pred, self.noise_variance)

        # 2. Invert (We want to minimize)
        loss = -torch.mean(sum_rate_batch)

        return loss
            
def benchmark(model, dataloader, P_total=1.0, noise_variance=1e-3, device='cpu'):
    """
    Evaluates the ML model agains optimization methods

    Inputs:
    model (torch.nn.Module): The ML model that will be tested
    dataloader (Dataloader): Test dataloader
    P_total (float): Total power budget for the system
    noise_variance (float): Sigma^2
    device: 'cuda' or 'cpu'

    Outputs:
    results (dict): Average sum-rate for each method
    """

    # 1. Set model to evaluation mode and move to device
    model.eval()
    model.to(device)

    # 2. Metrics list
    rates_zf_ml = []
    rates_mrt_ml = []
    rates_zf_wf = []
    rates_mrt_wf = []

    # 3. Testing loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 3.1 Move channels to device
            H = batch[0].to(device)
            tokens = batch[1].to(device)

            pred_weights = model(tokens)
            # Scaling
            ml_power = pred_weights * P_total
            # 3.2 Method 1: ML - MRT Directions + Model Power

            # Construct precoder
            V_mrt = get_mrt_directions(H)
            W_mrt_ml = V_mrt * torch.sqrt(ml_power.unsqueeze(-2))

            # Calculate rate
            rate_mrt_ml_batch = calculate_sum_rate(H, W_mrt_ml, noise_variance)
            rates_mrt_ml.append(rate_mrt_ml_batch.mean().item())

            # 3.3 Method 2: ML - ZF Directions + Model Power
            V_zf = get_zf_directions(H)
            W_zf_ml = V_zf *torch.sqrt(ml_power.unsqueeze(-2))

            rate_zf_ml_batch = calculate_sum_rate(H, W_zf_ml, noise_variance)
            rates_zf_ml.append(rate_zf_ml_batch.mean().item())

            # 3.3 Method 3: ZF + WF
            W_zf_raw = zf_precoder(H, P_total=P_total)
            W_zf_wf = apply_water_filling(W_zf_raw, H, P_total=P_total)

            rate_zf_batch = calculate_sum_rate(H, W_zf_wf, noise_variance)
            rates_zf_wf.append(rate_zf_batch.mean().item())

            # 3.4 Method 4: MRT + WF
            W_mrt_raw = mrt_precoder(H, P_total=P_total)
            W_mrt_wf = apply_water_filling(W_mrt_raw, H, P_total=P_total)

            rate_mrt_batch = calculate_sum_rate(H, W_mrt_wf, noise_variance)
            rates_mrt_wf.append(rate_mrt_batch.mean().item())

    results = {
        "ZF+ML": np.mean(rates_zf_ml),
        "MRT+ML": np.mean(rates_mrt_ml),
        "ZF+WF": np.mean(rates_zf_wf),
        "MRT+WF": np.mean(rates_mrt_wf)
    }

    return results