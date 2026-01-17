import torch
import torch.nn as nn
import torch.optim as optim

def generate_labels(channels, patch_cols, mode="proportional_fair", noise_variance: float=1e-3):
    """
    Generates ground truth labels for the carrier allocation task.

    Inputs:
        channels: Complex channel matrix - (Batch, Users, M, SC)
        patch_cols: The number of carriers in the allocation block
        mode: "greedy", "round_robin", "proportional_fair"
        noise_variance: Noise variance

    Outputs:
        labels: The winning user index - (Batch, Num_blocks)
    
    """
    B, K, M, SC = channels.shape
    num_blocks = SC // patch_cols
    device = channels.device

    # 1. Calculate block SNR
    mag_sq = torch.abs(channels)**2

    # Spatial average
    gain_freq = torch.mean(mag_sq, dim=2)

    gain_reshaped = gain_freq.view(B, K, num_blocks, patch_cols)
    block_gains = torch.mean(gain_reshaped, dim=3)

    block_snrs = block_gains / noise_variance

    if mode == "greedy":
        labels = torch.argmax(block_snrs, dim=1)
    elif mode == "round_robin":
        indices = torch.arange(num_blocks, device=device) % K
        labels = indices.expand(B, -1)
    elif mode == "proportional_fair":
        user_average_snr = torch.mean(block_snrs, dim=2, keepdim=True)

        pf_metric = block_snrs / (user_average_snr + 1e-8)

        labels = torch.argmax(pf_metric, dim=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return labels