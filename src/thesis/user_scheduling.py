import torch
import torch.nn as nn
import torch.optim as optim

class JointUtilityLoss(nn.Module):
    def __init__(self, alpha_entropy=0.01, alpha_power=0.001, noise_var=1e-9):
        super().__init__()
        self.alpha_entropy = alpha_entropy
        self.alpha_power = alpha_power
        self.noise_var = noise_var

    def forward(self, assignment_probs, power_values, channels):
        """
        Args:
            assignment_probs: (Batch, Users, Blocks) - Model Decision
            power_values:     (Batch, Users, Blocks) - Model Decision
            channels:         (Batch, Users, Ant, SC) - Real Physics
        """
        B, K, M, SC = channels.shape
        _, _, Blocks = assignment_probs.shape
        
        # 1. The Mapping Problem: Blocks -> Subcarriers
        # We need to stretch the model's decision to match the physical subcarriers
        subcarriers_per_block = SC // Blocks
        
        # Expand: (B, K, Blocks) -> (B, K, Blocks, SC_per_Block) -> (B, K, SC)
        A_full = assignment_probs.repeat_interleave(subcarriers_per_block, dim=2)
        P_full = power_values.repeat_interleave(subcarriers_per_block, dim=2) / subcarriers_per_block
        
        # 2. Physics: Handle Antennas (M)
        # Shape: (B, K, SC)
        channel_gains_full = torch.sum(torch.abs(channels)**2, dim=2)
        
        # 3. Calculate REAL SINR (Per Subcarrier)
        # Signal = P_sc * Gain_sc
        signal_power = P_full * channel_gains_full
        
        # Interference (Per Subcarrier)
        # Total power on this subcarrier from ALL users
        total_power_on_sc = torch.sum(signal_power, dim=1, keepdim=True)
        interference = total_power_on_sc - signal_power
        
        # SINR
        sinr = signal_power / (interference + self.noise_var)
        
        # 4. Calculate Rate (Per Subcarrier)
        # Rate = A * log(1 + SINR)
        weighted_rate = A_full * torch.log2(1 + sinr)
        
        # 5. Loss
        # Utility
        total_sum_rate = torch.sum(weighted_rate, dim=(1,2))
        loss_utility = -torch.mean(total_sum_rate)

        # Entropy
        entropy = -torch.sum(assignment_probs * torch.log(assignment_probs + 1e-8), dim=(1,2))
        loss_entropy = torch.mean(entropy)

        # Power regularization
        power_reg = torch.sum(power_values**2, dim=(1,2))
        loss_power = torch.mean(power_reg)
        
        return loss_utility + (self.alpha_entropy * loss_entropy) + (self.alpha_power * loss_power)