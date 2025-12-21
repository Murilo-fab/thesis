import numpy as np
from tqdm import tqdm
import torch

# Telecommunication imports
from .power_allocation import *

epsilon = 1e-18

class SumRateLoss(torch.nn.Module):
    """
    Criterion class for Sum Rate as Loss.

    It returns the sum rate as loss using the wireless channels and power

    Attributes:
    P_total (float): Total power for the system
    noise_variance (float): Sigma^2 
    """
    def __init__(self,
                 P_total: float = 1.0,
                 noise_variance: float = 1.0):
        """
        Constructor

        Inputs:
        P_total (float): Total power for the system
        noise_variance (float): Sigma^2 
        """
        super().__init__()
        self.P_total = P_total
        self.noise_variance = noise_variance

    def forward(self,
                pred_power: torch.Tensor,
                H: torch.Tensor) -> torch.Tensor:
        """
        Uses the negative sum rate as loss function
        
        Inputs
        pred_power (torch.tensor): Predicted power by the model [B, S, K]
        H (torch.tensor): Original channel [B, S, K, N]

        Output:
        loss (torch.tensor): negative sum rate
        """
        # 1. Scale power to comply with power budget
        power = pred_power * self.P_total

        # 2. Scale H such the E[|H|^2] approx 1
        avg_energy = torch.mean(torch.abs(H)**2, dim=(1, 2, 3), keepdim=True)
        scale_factor = 1.0 / torch.sqrt(avg_energy)

        H_norm = H * scale_factor
        
        # 3. Scale noise
        scaled_noise = self.noise_variance * (scale_factor**2)

        # 4. Construct directions using MRT precoder
        with torch.no_grad():
            V_mrt = get_mrt_directions(H_norm)

        # 5. Combine power and directions
        W_pred = V_mrt * torch.sqrt(power.unsqueeze(-2) + epsilon)

        # 6. Calculate sum rate
        sum_rate_batch = calculate_sum_rate(H_norm, W_pred, scaled_noise)

        # 7. Invert (We want to minimize)
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
    def __init__(self,
                 P_total: float = 1.0,
                 noise_variance: float = 1.0):
        """
        Constructor

        Inputs:
        P_total (float): Total power for the system
        noise_variance (float): Sigma^2 
        """
        super().__init__()
        self.P_total = P_total
        self.noise_variance = noise_variance

    def forward(self,
                W_pred: torch.Tensor,
                H: torch.Tensor) -> torch.Tensor:
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
            
class Benchmark:
    def __init__(self,
                 P_total: float = 1.0,
                 noise_variance: float = 1e-3,
                 min_rate_threshold: float = 0.1):
        
        self.P_total = P_total
        self.noise_variance = noise_variance
        self.min_rate_threshold = min_rate_threshold
    
    def calculate_metrics(self, H, P_alloc):
        """
        
        """
        # Get ZF directions
        V_zf = get_zf_directions(H)


def benchmark(model: torch.nn.Module,
              dataloader,
              P_total: float = 1.0,
              noise_variance: float = 1e-3,
              device: torch.device = 'cpu') -> dict:
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
    model.to(device)
    model.eval()

    # 2. Metrics list
    rates_zf_ml = []
    rates_mrt_ml = []
    rates_zf_wf = []
    rates_mrt_wf = []

    # Power Trackers
    power_usage_ml = []
    power_usage_wf = []

    # 3. Testing loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 3.1 Move channels to device
            H_raw = batch[0].to(device)
            tokens = batch[1].to(device)

            # Scaling
            # 3.2 Calculate the normalization factor based on the current batch
            # We want E[|h|^2] to be approx 1.0
            avg_power = torch.mean(torch.abs(H_raw)**2)
            norm_factor = 1.0 / torch.sqrt(avg_power)

            # 3.3 Normalize H
            H = H_raw * norm_factor

            # 3.4 IMPORTANT: Scale the Noise Variance
            # If you boost signal by X^2, you must boost noise by X^2 to keep SNR the same.
            # scale = norm_factor^2
            scaled_noise_variance = noise_variance * (norm_factor**2).item()

            # ML Prediction
            pred_weights = model(tokens)
            ml_power = pred_weights * P_total

            # ML Power Check
            current_ml_power = torch.sum(ml_power, dim=(1, 2))
            power_usage_ml.append(current_ml_power.mean().item())
            
            # 3.5 Method 1: ML - MRT Directions + Model Power

            # Construct precoder
            V_mrt = get_mrt_directions(H)
            W_mrt_ml = V_mrt * torch.sqrt(ml_power.unsqueeze(-2))

            # Calculate rate
            rate_mrt_ml_batch = calculate_sum_rate(H, W_mrt_ml, scaled_noise_variance)
            rates_mrt_ml.append(rate_mrt_ml_batch.mean().item())

            # 3.6 Method 2: ML - ZF Directions + Model Power
            V_zf = get_zf_directions(H)
            W_zf_ml = V_zf *torch.sqrt(ml_power.unsqueeze(-2))

            rate_zf_ml_batch = calculate_sum_rate(H, W_zf_ml, scaled_noise_variance)
            rates_zf_ml.append(rate_zf_ml_batch.mean().item())

            # 3.7 Method 3: ZF + WF
            W_zf_raw = zf_precoder(H, P_total=P_total)
            W_zf_wf = apply_water_filling(W_zf_raw, H, P_total=P_total, noise_variance=scaled_noise_variance)

            # ZF + WF Power check
            # Power is sum of squared magnitude of beamforming vectors
            # W shape: [B, S, N, K]
            wf_power_per_stream = torch.sum(torch.abs(W_zf_wf)**2, dim=-2) # Sum over Antennas -> [B, S, K]
            current_wf_power = torch.sum(wf_power_per_stream, dim=(1, 2))      # Sum over S and K -> [B]
            power_usage_wf.append(current_wf_power.mean().item())

            rate_zf_batch = calculate_sum_rate(H, W_zf_wf, scaled_noise_variance)
            rates_zf_wf.append(rate_zf_batch.mean().item())

            # 3.8 Method 4: MRT + WF
            W_mrt_raw = mrt_precoder(H, P_total=P_total)
            W_mrt_wf = apply_water_filling(W_mrt_raw, H, P_total=P_total, noise_variance=scaled_noise_variance)

            rate_mrt_batch = calculate_sum_rate(H, W_mrt_wf, scaled_noise_variance)
            rates_mrt_wf.append(rate_mrt_batch.mean().item())

    # --- PRINT POWER REPORT ---
    avg_ml = np.mean(power_usage_ml)
    avg_wf = np.mean(power_usage_wf)
    
    print("\n" + "="*40)
    print("POWER CONSUMPTION REPORT")
    print("="*40)
    print(f"Target P_total: {P_total:.4f}")
    print(f"Actual ML Power: {avg_ml:.4f}")
    print(f"Actual WF Power: {avg_wf:.4f}")
    
    if abs(avg_ml - avg_wf) > 1e-2:
        print("WARNING: Power Mismatch detected!")
    else:
        print("SUCCESS: Power budgets are matched.")
    print("="*40 + "\n")

    results = {
        "ZF+ML": np.mean(rates_zf_ml),
        "MRT+ML": np.mean(rates_mrt_ml),
        "ZF+WF": np.mean(rates_zf_wf),
        "MRT+WF": np.mean(rates_mrt_wf)
    }

    return results