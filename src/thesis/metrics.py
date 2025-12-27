import numpy as np
from tqdm import tqdm
from typing import Literal

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


class MSELoss(torch.nn.Module):
    """

    """
    def __init__(self,
                 P_total: float = 1.0,
                 noise_variance: float = 1.0,
                 precoder: Literal["mrt", "zf"] = "zf"):
        """
        Constructor

        Inputs:
            P_total (float): Total power for the system
            noise_variance (float): Sigma^2 
        """
        super().__init__()
        self.P_total = P_total
        self.noise_variance = noise_variance
        self.precoder = precoder

    def forward(self,
                pred_power: torch.Tensor,
                H: torch.Tensor) -> torch.Tensor:
        """
        """
        # 2. Construct directions using precoder
        with torch.no_grad():
            if self.precoder == "zf":
                W = zf_precoder(H, self.P_total)
            elif self.precoder == "mrt":
                W = mrt_precoder(H, self.P_total)
            else:
                raise ValueError(f"Loss expected a precoder configuration")

        W_opt = apply_water_filling(W, H, self.P_total, noise_variance=self.noise_variance)

        W_sq = torch.abs(W_opt) ** 2

        opt_power = torch.sum(W_sq, dim=-2)

        loss = torch.nn.functional.mse_loss(pred_power * self.P_total, opt_power)

        return loss


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
                 noise_variance: float = 1.0,
                 precoder: Literal["mrt", "zf"] = "zf"):
        """
        Constructor

        Inputs:
            P_total (float): Total power for the system
            noise_variance (float): Sigma^2 
        """
        super().__init__()
        self.P_total = P_total
        self.noise_variance = noise_variance
        self.precoder = precoder

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

        # 2. Construct directions using precoder
        with torch.no_grad():
            if self.precoder == "zf":
                V = get_zf_directions(H)
            elif self.precoder == "mrt":
                V = get_mrt_directions(H)
            else:
                raise ValueError(f"Loss expected a precoder configuration")

        # 3. Combine power and directions
        W_pred = V * torch.sqrt(power.unsqueeze(-2) + epsilon)

        # 4. Calculate sum rate
        sum_rate_batch, _ = calculate_sum_rate(H, W_pred, self.noise_variance)

        # 5. Invert (We want to minimize)
        loss = -torch.mean(sum_rate_batch)

        return loss
    
class SumRateLossWithQoS(torch.nn.Module):
    """
    Criterion class for Sum Rate with QoS as Loss.

    It returns the sum rate as loss using the wireless channels and power

    Attributes:
        P_total (float): Total power for the system
        noise_variance (float): Sigma^2 
    """
    def __init__(self,
                 P_total: float = 1.0,
                 noise_variance: float = 1.0,
                 min_rate_threshold = 0.5,
                 penalty_weight = 10.0,
                 precoder: Literal["mrt", "zf"] = "zf"):
        """
        Constructor

        Inputs:
            P_total (float): Total power for the system
            noise_variance (float): Sigma^2 
        """
        super().__init__()
        self.P_total = P_total
        self.noise_variance = noise_variance
        self.min_rate_threshold = min_rate_threshold
        self.penalty_weight = penalty_weight
        self.precoder = precoder

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

        # 2. Construct directions using precoder
        with torch.no_grad():
            if self.precoder == "zf":
                V = get_zf_directions(H)
            elif self.precoder == "mrt":
                V = get_mrt_directions(H)
            else:
                raise ValueError(f"Loss expected a precoder configuration")

        # 3. Combine power and directions
        W_pred = V * torch.sqrt(power.unsqueeze(-2) + epsilon)

        # 4. Calculate sum rate
        sum_rate_batch, rates = calculate_sum_rate(H, W_pred, self.noise_variance)

        # 5. Calculate user rates
        user_rates = torch.mean(rates, dim=1)

        # 6. QoS violations
        violation = torch.relu(self.min_rate_threshold - rates)
        penalty = torch.mean(torch.sum(violation ** 2, dim=1))

        # 5. Invert (We want to minimize)
        loss = -torch.mean(sum_rate_batch) + self.penalty_weight * penalty

        return loss

class EmbeddingAnalyzer:
    def __init__(self, model, device="cuda"):
        """
        Analyzer for visualizing the internal physics-alignment of the model.
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def run(self, dataloader, num_samples=2000, perplexity=30, results_dir=None):
        """
        Main execution method.
        1. Extracts data.
        2. Computes t-SNE.
        3. Plots the 3-panel analysis.
        """
        print(f"1. Extracting embeddings for {num_samples} samples...")
        physics, embeddings, power = self._extract_data(dataloader, num_samples)
        
        print(f"2. Running t-SNE on {len(embeddings)} vectors...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        z_tsne = tsne.fit_transform(embeddings)
        
        print("3. Generating Plots...")
        fig = self._plot_analysis(physics, power, z_tsne)

        if results_dir:
            fig.savefig(
                f"{results_dir}/embeddings.png",
                dpi=300,
                bbox_inches="tight"
                )
            
        return fig

    def _extract_data(self, dataloader, num_samples):
        """
        Internal helper to run inference and gather numpy arrays.
        """
        physics_list = []
        embeddings_list = [] 
        power_list = [] 
        
        collected = 0
        
        with torch.no_grad():
            # Note: Standard order is usually (tokens, channels). 
            # If your loader is different, swap these variables.
            for channels, tokens in dataloader:
                if collected >= num_samples: break
                
                tokens = tokens.to(self.device)     # [B, K, S, F]
                channels = channels.to(self.device) # [B, S, K, M]
                
                B, K, S, F = tokens.shape

                # A. Physics (Energy per user)
                # H: [B, S, K, M] -> |h|^2 -> Sum M (Antennas) -> Mean S (Subcarriers)
                # Result: [B, K]
                raw_energy = torch.mean(torch.sum(torch.abs(channels)**2, dim=3), dim=1)
                
                # B. Enbeddings
                x_flat = tokens.view(B*K, S, F)
                enc_out, _ = self.model.encoder(x_flat) # [B*K, S, D]
                
                # [B*K, S, D] -> Take CLS (idx 0)
                user_embeddings = enc_out[:, 0, :] 
                
                # C. Prediction (Power)
                pred_p = self.model(tokens) # [B, S, K]
                
                # Mean over Subcarriers -> [B, K]
                user_power = torch.mean(pred_p, dim=1)
                
                # D. Flatten and store
                physics_list.append(raw_energy.flatten().cpu().numpy())
                embeddings_list.append(user_embeddings.cpu().numpy())
                power_list.append(user_power.flatten().cpu().numpy())
                
                collected += B * K

        return (np.concatenate(physics_list), 
                np.vstack(embeddings_list), 
                np.concatenate(power_list))

    def _plot_analysis(self, physics, power, z_tsne):
        """
        Generates the visualization with Log-Scale fix.
        """
        # 1. Transform Physics to dB (Handling zeroes safely)
        physics_db = 10 * np.log10(physics + 1e-16)

        # 2. Clip extreme outliers for better color contrast
        vmin = np.percentile(physics_db, 5) 
        vmax = np.percentile(physics_db, 95)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Latent Space (Colored by dB) 
        sc1 = axes[0].scatter(z_tsne[:, 0], z_tsne[:, 1], c=physics_db, 
                              cmap='viridis', s=10, alpha=0.7, vmin=vmin, vmax=vmax)
        axes[0].set_title("1. Latent Space\n(Colored by Channel dB)")
        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")
        plt.colorbar(sc1, ax=axes[0], label="Gain (dB)")

        # Plot 2: Decision Space (Colored by Power) 
        sc2 = axes[1].scatter(z_tsne[:, 0], z_tsne[:, 1], c=power, 
                              cmap='plasma', s=10, alpha=0.7)
        axes[1].set_title("2. Decision Space\n(Colored by Power)")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_yticks([])
        plt.colorbar(sc2, ax=axes[1], label="Power")

        # Plot 3: End-to-End (Log-Scale X-Axis) 
        # Goal: Unravel the "Vertical Wall" to see the Transfer Function
        sc3 = axes[2].scatter(physics_db, power, c=z_tsne[:, 0], 
                              cmap='twilight', s=10, alpha=0.6)
        axes[2].set_title("3. Transfer Function\n(Log-Domain Physics)")
        axes[2].set_xlabel("Channel Gain (dB)")
        axes[2].set_ylabel("Predicted Power")
        axes[2].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.show()
        
        return fig

class Benchmark:
    def __init__(self,
                 dataloader,
                 P_total: float = 1.0,
                 noise_variance: float = 1e-3,
                 min_rate_threshold: float = 2.0,
                 precoder: Literal["mrt", "zf"] = "mrt",
                 device: str = "cpu"):
        
        self.dataloader = dataloader
        self.P_total = P_total
        self.noise_variance = noise_variance
        self.min_rate_threshold = min_rate_threshold
        self.precoder = precoder
        self.device = device
    
    def evaluate_model(self, model: torch.nn.Module):
        model.to(self.device)
        model.eval()

        all_sum_rates = []
        all_fairness = []
        total_outage_count = 0
        total_samples = 0

        with torch.no_grad():
            for channels, tokens in tqdm(self.dataloader):
                # Move batch to device
                channels, tokens = channels.to(self.device), tokens.to(self.device)

                P = model(tokens)
                P = P.unsqueeze(-2)

                if self.precoder == "mrt":
                    V = get_mrt_directions(channels)
                elif self.precoder == "zf":
                    V = get_zf_directions(channels)
                else:
                    raise ValueError("The Benchmark module needs a precoder")
                
                W = V * torch.sqrt(P)
                batch_results = self.calculate_metrics(channels, W)

                all_sum_rates.append(batch_results['raw_sum_rates'])
                all_fairness.append(batch_results['avg_fairness'])
                total_outage_count += batch_results['outage_count']
                total_samples += batch_results['total_users']

        flat_rates = np.concatenate(all_sum_rates)

        return {
            "avg_sum_rate": np.mean(flat_rates),
            "avg_fairness": np.mean(all_fairness),
            "outage_prob": total_outage_count / total_samples,
            "raw_sum_rates": flat_rates # For Plotting
        }

    def evaluate_wmmse(self):
        all_sum_rates = []
        all_fairness = []
        total_outage_count = 0
        total_samples = 0

        with torch.no_grad():
            for channels, tokens in tqdm(self.dataloader):
                # Move batch to device
                channels, tokens = channels.to(self.device), tokens.to(self.device)

                _, W = wmmse_solver(channels, self.P_total, self.noise_variance)
                batch_results = self.calculate_metrics(channels, W)

                all_sum_rates.append(batch_results['raw_sum_rates'])
                all_fairness.append(batch_results['avg_fairness'])
                total_outage_count += batch_results['outage_count']
                total_samples += batch_results['total_users']

        flat_rates = np.concatenate(all_sum_rates)

        return {
            "avg_sum_rate": np.mean(flat_rates),
            "avg_fairness": np.mean(all_fairness),
            "outage_prob": total_outage_count / total_samples,
            "raw_sum_rates": flat_rates # For Plotting
        }

    def evaluate_zf_epa(self):
        all_sum_rates = []
        all_fairness = []
        total_outage_count = 0
        total_samples = 0


        with torch.no_grad():
            for channels, tokens in tqdm(self.dataloader):
                # Move batch to device
                channels, tokens = channels.to(self.device), tokens.to(self.device)

                B, S, K, N = channels.shape
                P = torch.ones(B, S, K, device=self.device) / (K * S)
                P = P.unsqueeze(-2)

                V = get_zf_directions(channels)
                W = V * torch.sqrt(P)

                batch_results = self.calculate_metrics(channels, W)

                all_sum_rates.append(batch_results['raw_sum_rates'])
                all_fairness.append(batch_results['avg_fairness'])
                total_outage_count += batch_results['outage_count']
                total_samples += batch_results['total_users']

        flat_rates = np.concatenate(all_sum_rates)

        return {
            "avg_sum_rate": np.mean(flat_rates),
            "avg_fairness": np.mean(all_fairness),
            "outage_prob": total_outage_count / total_samples,
            "raw_sum_rates": flat_rates # For Plotting
        }

    def evaluate_zf_wf(self):
        all_sum_rates = []
        all_fairness = []
        total_outage_count = 0
        total_samples = 0


        with torch.no_grad():
            for channels, tokens in tqdm(self.dataloader):
                # Move batch to device
                channels, tokens = channels.to(self.device), tokens.to(self.device)

                W = zf_precoder(channels, self.P_total)
                W_opt = apply_water_filling(W, channels, self.P_total, self.noise_variance)
                batch_results = self.calculate_metrics(channels, W_opt)

                all_sum_rates.append(batch_results['raw_sum_rates'])
                all_fairness.append(batch_results['avg_fairness'])
                total_outage_count += batch_results['outage_count']
                total_samples += batch_results['total_users']

        flat_rates = np.concatenate(all_sum_rates)

        return {
            "avg_sum_rate": np.mean(flat_rates),
            "avg_fairness": np.mean(all_fairness),
            "outage_prob": total_outage_count / total_samples,
            "raw_sum_rates": flat_rates # For Plotting
        }
    
    def evaluate_mrt_epa(self):
        all_sum_rates = []
        all_fairness = []
        total_outage_count = 0
        total_samples = 0


        with torch.no_grad():
            for channels, tokens in tqdm(self.dataloader):
                # Move batch to device
                channels, tokens = channels.to(self.device), tokens.to(self.device)

                B, S, K, N = channels.shape
                P = torch.ones(B, S, K, device=self.device) / (K * S)
                P = P.unsqueeze(-2)
                
                V = get_mrt_directions(channels)
                W = V * torch.sqrt(P)

                batch_results = self.calculate_metrics(channels, W)

                all_sum_rates.append(batch_results['raw_sum_rates'])
                all_fairness.append(batch_results['avg_fairness'])
                total_outage_count += batch_results['outage_count']
                total_samples += batch_results['total_users']

        flat_rates = np.concatenate(all_sum_rates)

        return {
            "avg_sum_rate": np.mean(flat_rates),
            "avg_fairness": np.mean(all_fairness),
            "outage_prob": total_outage_count / total_samples,
            "raw_sum_rates": flat_rates # For Plotting
        }

    def evaluate_mrt_wf(self):
        all_sum_rates = []
        all_fairness = []
        total_outage_count = 0
        total_samples = 0


        with torch.no_grad():
            for channels, tokens in tqdm(self.dataloader):
                # Move batch to device
                channels, tokens = channels.to(self.device), tokens.to(self.device)

                W = mrt_precoder(channels, self.P_total)
                W_opt = apply_water_filling(W, channels, self.P_total, self.noise_variance)
                batch_results = self.calculate_metrics(channels, W_opt)

                all_sum_rates.append(batch_results['raw_sum_rates'])
                all_fairness.append(batch_results['avg_fairness'])
                total_outage_count += batch_results['outage_count']
                total_samples += batch_results['total_users']

        flat_rates = np.concatenate(all_sum_rates)

        return {
            "avg_sum_rate": np.mean(flat_rates),
            "avg_fairness": np.mean(all_fairness),
            "outage_prob": total_outage_count / total_samples,
            "raw_sum_rates": flat_rates # For Plotting
        }

    def evaluate(self, model: torch.nn.Module):
        """
        Evaluate the Model

        Inputs:
            model (nn.Module)

        Outputs:
            results (dict)
        """
        results = {}

        results["Model"] = self.evaluate_model(model)
        results["MRT + EPA"] = self.evaluate_mrt_epa()
        results["MRT + WF"] = self.evaluate_mrt_wf()
        results["ZF + EPA"] = self.evaluate_zf_epa()
        results["ZF + WF"] = self.evaluate_zf_wf()
        results["WMMSE"] = self.evaluate_wmmse()

        return results
    
    def calculate_metrics(self, H, W):
        # 1. Calculate sum-rate
        sum_rate, rates = calculate_sum_rate(H, W, self.noise_variance) # Shape: [B, S], [B, S, K]
        user_rates = torch.sum(rates, dim=1) # Shape: [B, K]

        # 2. Metric 1: Sum-Rate per sample
        sum_rate_per_sample = torch.mean(sum_rate, dim=1) # Shape [B,]

        # 3. Metric 2: Jain's Fairness
        # J = (sum x)^2 / (N * sum x^2)
        num_users = user_rates.shape[1]
        num = torch.sum(user_rates, dim=1)**2
        denom = num_users * torch.sum(user_rates**2, dim=1) + epsilon
        J_index_per_sample = num / denom

        # 4. Metric 3: Outage probability
        # P(user rate < rate_threshold)
        outage_count = (user_rates < self.min_rate_threshold).float().sum()
        total_users = user_rates.numel()

        return {
            "avg_sum_rate": torch.mean(sum_rate_per_sample).item(),
            "avg_fairness": torch.mean(J_index_per_sample).item(),
            "outage_count": outage_count,
            "total_users": total_users,
            "raw_sum_rates": sum_rate_per_sample.detach().cpu().numpy(), # For plotting CDF
        }
    
    def plot_cdf(self, methods_dict, results_dir=None):
        """
        methods_dict: {'AI Model': results_dict, 'Equal Power': results_dict}
        """
        plt.figure(figsize=(10,6))
        for name, results in methods_dict.items():
            data = np.sort(results['raw_sum_rates'])
            yvals = np.arange(len(data)) / float(len(data) - 1)
            plt.plot(data, yvals, linewidth=2.5, label=f"{name} (Avg: {results['avg_sum_rate']:.2f})")

        plt.title("Sum-Rate Capacity CDF", fontsize=14)
        plt.xlabel("Sum-Rate Capacity (bps/Hz)", fontsize=12)
        plt.ylabel("Cumulative Probability", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if results_dir:
            plt.savefig(
                f"{results_dir}/sum-rate.png",
                dpi=300,
                bbox_inches="tight"
                )
            
        plt.show()