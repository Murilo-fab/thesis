import numpy as np
from typing import Literal

import seaborn as sns
import matplotlib.pyplot as plt

import torch

# Telecommunication imports
from thesis.power_allocation import *

epsilon = 1e-18

class DatasetVisualizer:
    """
    """
    def __init__(self, dataset_H):
        """
        """
        # 1. Ensure we are working with Numpy
        if torch.is_tensor(dataset_H):
            self.H = dataset_H.detach().cpu().numpy()
        else:
            self.H = dataset_H
            
        self.num_samples, self.K, self.M, self.S = self.H.shape

    def _calculate_spatial_correlation(self):
        """
        Helper: Calculates pairwise correlation between users in the same group.
        """
        correlations = []
        # 1. Look at the first subcarrier
        # Shape: [B, K, N]
        H_spatial = self.H[:, :, :, 0]
        
        # 2. Normalize vectors to unit norm
        norms = np.linalg.norm(H_spatial, axis=2, keepdims=True)
        H_norm = H_spatial / (norms + epsilon)

        for b in range(self.num_samples):
            # 3.1 Matrix of users (K, M)
            h_b = H_norm[b]
            # 3.2 Gram Matrix (K, K) -> entries are |h_i * h_j'|
            gram = np.abs(h_b @ h_b.conj().T)
            # 3.3 Extract off-diagonal elements (cross-correlations)
            mask = ~np.eye(self.K, dtype=bool)
            corr_values = gram[mask]
            correlations.extend(corr_values)

        return np.array(correlations)
    
    def _calculate_condition_numbers(self):
        """Helper: Calculates Condition Number (Kappa) of the channel matrices."""
        # Condition number of H (K x M) matrix
        # Kappa = max_singular_value / min_singular_value
        # High Kappa = Ill-conditioned (Hard for Zero-Forcing)
        
        # 1. Look the first subcarrier
        # Shape: [B, K, N] for SVD
        H_spatial = self.H[:, :, :, 0] 
        
        # 2. Compute singular values (S)
        # svd returns S ordered from large to small
        _, S, _ = np.linalg.svd(H_spatial) 
        
        # 3. Calculate condition numbers
        # S is shape (Samples, K) (assuming K < M)
        max_sv = S[:, 0]
        min_sv = S[:, -1]
        
        cond_nums = max_sv / (min_sv + epsilon)
        
        # 4. Convert to dB scale for easier plotting
        return 20 * np.log10(cond_nums)

    def _calculate_gain_imbalance(self):
        """Helper: Ratio of strongest user to weakest user in dB."""
        # 1. Power per user: |h|^2
        # Get the first subcarrier
        powers = np.linalg.norm(self.H[:, :, :, 0], axis=2)**2
        
        max_p = np.max(powers, axis=1)
        min_p = np.min(powers, axis=1)
        
        ratios_linear = max_p / (min_p + epsilon)
        # 2. Convert to dB scale for easier plotting
        return 10 * np.log10(ratios_linear)

    def plot_dataset_stats(self, results_dir: str = None):
        """
        Generates a 1x3 dashboard of dataset difficulty metrics.
        """
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Condition Number Distribution ---
        kappas = self._calculate_condition_numbers()
        sns.histplot(kappas, bins=30, kde=True, ax=axes[0], color='teal')
        axes[0].set_title(f"Channel Condition Number (dB)\nAvg: {np.mean(kappas):.2f} dB")
        axes[0].set_xlabel("Condition Number (dB)")
        axes[0].set_ylabel("Count")
        # Add Reference Lines
        axes[0].axvline(10, color='r', linestyle='--', label='Well Conditioned (<10dB)')
        axes[0].legend()

        # 2. Spatial Correlation Distribution ---
        corrs = self._calculate_spatial_correlation()
        sns.histplot(corrs, bins=30, kde=True, ax=axes[1], color='orange')
        axes[1].set_title(f"User Spatial Correlation\nAvg: {np.mean(corrs):.2f}")
        axes[1].set_xlabel("Cosine Similarity |h_i @ h_j|")
        axes[1].set_xlim(0, 1)

        # 3. Near-Far Gain Ratio ---
        ratios = self._calculate_gain_imbalance()
        sns.histplot(ratios, bins=30, kde=True, ax=axes[2], color='purple')
        axes[2].set_title(f"Near-Far Gain Ratio (dB)\nAvg: {np.mean(ratios):.2f} dB")
        axes[2].set_xlabel("Max/Min Power Ratio (dB)")
        
        plt.tight_layout()

        if results_dir:
            fig.savefig(
                f"{results_dir}/dataset_stats.png",
                dpi=300,
                bbox_inches="tight"
                )

        plt.show()


    def plot_channel_response(self, sample_idx=0, user_idx=0, results_dir: str = None):
        """
        Visualizes the Frequency Selectivity (The "Wiggles") for a specific user.
        """
        # H shape: [B, K, N, SC]
        # 1. Extract specific user
        h_user = self.H[sample_idx, user_idx, :, :] # [N, SC]
        
        # 2. Compute Magnitude (Power) response
        # Shape: [N, SC]
        power_response = np.abs(h_user)**2
        
        plt.figure(figsize=(10, 5))
        
        # 3. Plot Heatmap
        sns.heatmap(power_response, cmap='viridis', cbar_kws={'label': 'Power'})
        
        plt.title(f"Channel Magnitude Response (Sample {sample_idx}, User {user_idx})")
        plt.xlabel("Subcarrier Index (Frequency)")
        plt.ylabel("Antenna Index (Space)")

        if results_dir:
            plt.savefig(
                f"{results_dir}/channel_response.png",
                dpi=300,
                bbox_inches="tight"
                )
            
        plt.show()

