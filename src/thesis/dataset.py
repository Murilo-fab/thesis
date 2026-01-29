# General imports
import numpy as np
from tqdm import tqdm

# Torch imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Telecommunication imports
import DeepMIMOv3
from thesis.scenario_props import *
from thesis.utils import get_parameters

import warnings
warnings.filterwarnings('ignore')

# Configuration Constants
DEFAULT_NUM_UE_ANTENNAS = 1
DEFAULT_SUBCARRIER_SPACING = 30e3  # Hz
DEFAULT_BS_ROTATION = np.array([0, 0, -135])  # (x, y, z) degrees
DEFAULT_NUM_PATHS = 20

class DeepMIMOGenerator:
    """
    Represents a DeepMIMO dataset generator

    Attributes:
        scenario_name (str): The scenario name used in the dataset
        bs_idx (int): The index of the active base station
        scenario_folder (str): Path to to the directory with the scenarios
        params (dict): A dict with DeepMIMO parameters
        all_channels (np.array): Array with all channels in the scenario [K, N, SC]
        num_total_users (int): Total number of users in the dataset [K]
        user_gains (np.array): Gains of all users [K,]
        h_spatial (np.array): Normalized center carrier of each user [K, N]
    """
    def __init__(self,
                 scenario_name: str = 'city_6_miami',
                 scale_factor: float = 1e6):
        """
        Initialize the DeepMIMO dataset generator.
        Generates the channels, gains and normalized center carrier.
        
        Inputs:
            scenario_name (str): The name of the selected scenario
            bs_idx (int): The index of the active base station
            scenario_folder (str): Path to to the directory with the scenarios
        """
        self.scenario_name = scenario_name
        self.scale_factor = scale_factor

        self.params = get_parameters(scenario_name)
        self.n_subcarriers = self.params['OFDM']['subcarriers']
        self.n_ant_bs = self.params['bs_antenna']['shape'][0]

        self.all_channels, self.num_total_users = self.get_channels()
        self.user_gains, self.h_spatial = self.get_matrices()
    
    def get_channels(self) -> tuple[np.ndarray, int]:
        """
        Generates the all channels using DeepMIMO.
        Removes users without a path to the BS.
        Removes the user antenna dimension, since all scenarios are generated with only one antenna in the UE.

        Inputs:

        Outputs:
            all_channels (np.ndarray): Array with all channels in the scenario [K, N, SC]
            num_total_users (int): Total number of users in the dataset [K]
        """
        # 1. Users DeepMIMO to generated the data
        deepmimo_data = DeepMIMOv3.generate_data(self.params)

        # 2. Removes users without a path to the BS
        idxs = np.where(deepmimo_data[0]['user']['LoS'] != -1)[0]
        cleaned_deepmimo_data = deepmimo_data[0]['user']['channel'][idxs]
        # 3. Removes the UE antenna dimension
        all_channels = cleaned_deepmimo_data.squeeze() * self.scale_factor

        num_total_users = all_channels.shape[0]

        return all_channels, num_total_users
    
    def get_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates the gains and spatial signatures matrices
        
        Inputs:

        Outputs:
            user_gains (np.ndarray): Gains of all users [K,]
            h_spatial (np.ndarray): Normalized center carrier of each user [K, N]
        """
        # 1. Gains (Average Power over subcarriers)
        user_gains = np.linalg.norm(self.all_channels, axis=(1, 2))**2 / self.n_subcarriers

        # 2. Spatial Signatures (Normalized Center Subcarrier)
        mid_sub = self.n_subcarriers // 2
        h_spatial_raw = self.all_channels[:, :, mid_sub]
        norms = np.linalg.norm(h_spatial_raw, axis=1, keepdims=True)
        h_spatial = h_spatial_raw / (norms + 1e-9)

        return user_gains, h_spatial

    def get_valid_mask_for_user(self,
                                target_user_idx: int,
                                min_corr: float,
                                max_corr: float,
                                max_gain_ratio: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a mask with the valid users based on gain and correlation conditions

        Inputs:
            target_iser_idx (int): The index of the compared user
            min_corr (float): Minimum correlation between users
            max_corr (float): Maximum correlation beween users
            max_gain_ratio (float): Maximum gain ratio between users

        Outputs:
            mask_corr (np.ndarray): The mask of users with valid correlation [K,]
            mask_gain (np.ndarray): The mask of users with valid gain [K,]
        """
        # 1. Vectorized Correlation
        target_vec = self.h_spatial[target_user_idx]
        corrs = np.abs(self.h_spatial @ target_vec.conj())
        mask_corr = (corrs >= min_corr) & (corrs <= max_corr)

        # 2. Vectorized Gain Ratio 
        target_gain = self.user_gains[target_user_idx]
        all_gains = self.user_gains

        g_min = np.minimum(target_gain, all_gains) + 1e-9
        g_max = np.maximum(target_gain, all_gains)
        ratios = g_max / g_min
        mask_gain = ratios <= max_gain_ratio

        return mask_corr & mask_gain
    
    def generate_dataset(self,
                         num_samples: int,
                         num_users: int,
                         min_corr: float = 0.5,
                         max_corr: float = 0.9,
                         max_gain_ratio: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample users and generate a dataset with minimum and maximum correlation between users
        and a maximum gain ratio between users.

        Inputs:
            num_samples (int): Number of samples in the final dataset
            num_users (int): Number of users in each sample
            min_corr (float): Minimum correlation between users
            max_corr (float): Maximum correlation beween users
            max_gain_ratio (float): Maximum gain ratio between users

        Outputs:
            dataset_H (np.array): The final channel dataset [B, K, N, SC]
            indices (np.array): The indices of selected users [B, K]
        """
        dataset_indices = []
        pbar = tqdm(total=num_samples, desc="Generating Scenarios")

        attempts = 0
        # 1. Keeps running while the number of samples is not achieved
        while len(dataset_indices) < num_samples:
            # 2. Random Start
            current_group = [np.random.randint(0, self.num_total_users - 1)]
            # 3. Init Mask (All users valid except self)
            candidate_mask = np.ones(self.num_total_users, dtype=bool)
            candidate_mask[current_group[0]] = False

            group_failed = False
            # 4. Greedily add users
            for _ in range(num_users - 1):
                last_added = current_group[-1]

                # 5. Update candidates based on compatibility with the newest member
                step_mask = self.get_valid_mask_for_user(last_added, min_corr, max_corr, max_gain_ratio)
                candidate_mask &= step_mask
                # 6. Get valid indices and check if the group is possible
                valid_indices = np.where(candidate_mask)[0]
                if len(valid_indices) == 0:
                    group_failed = True
                    break
                # 7. Randomly selects a new user
                next_user = np.random.choice(valid_indices)
                current_group.append(next_user)
                candidate_mask[next_user] = False
            # 8. Adds the new group of users to dataset
            if not group_failed:
                dataset_indices.append(current_group)
                pbar.update(1)
            else:
                attempts += 1
        # 9. Gets the channels from the indices
        pbar.close()
        indices = np.array(dataset_indices)
        dataset_H = self.all_channels[indices] # Shape: (Samples, Users, Antennas, Subcarriers)

        return dataset_H
    

def generate_combined_dataset(scenario_name, n_samples_total, K):
    """
    Step 1: Generate Easy + Hard datasets and concatenate.
    """
    # Initialize your generator
    gen = DeepMIMOGenerator(scenario_name)
    
    n_half = n_samples_total // 2
    
    print(f"Generating {n_samples_total} samples (Half Easy / Half Hard)...")
    
    # 1. Easy Dataset (Orthogonal, Balanced)
    H_easy = gen.generate_dataset(
        num_samples=n_half, num_users=K,
        min_corr=0.0, max_corr=0.4, max_gain_ratio=10.0
    )
    
    # 2. Hard Dataset (Interfering, Unbalanced)
    H_hard = gen.generate_dataset(
        num_samples=n_half, num_users=K,
        min_corr=0.6, max_corr=0.9, max_gain_ratio=50.0
    )
    
    # 3. Concatenate and Shuffle
    H_all = np.concatenate([H_easy, H_hard], axis=0)
    np.random.shuffle(H_all)
    
    
    return torch.tensor(H_all)

import os
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from thesis.utils import build_model_from_config


import torch

def calculate_noise(H_raw, target_snr_db=5.0, p_total=1.0):
    """
    Prepares channel tensors and calculates the specific noise level required 
    to achieve the target average SNR across the dataset.
    
    Args:
        H_raw (np.ndarray): Raw channel data (N, K, Tx, Sub)
        target_snr_db (float): Desired average SNR in dB
        p_total (float): Total transmit power (usually normalized to 1.0)
        device (str): 'cuda' or 'cpu'
        
    Returns:
        H_tensor (torch.Tensor): Complex channel tensor on device
        noise_val (float): The calculated noise power (sigma^2)
    """
    # 2. Calculate Average Signal Power
    avg_channel_gain = torch.mean(torch.abs(H_raw)**2)
    signal_power = p_total * avg_channel_gain
    
    # 3. Calculate Required Noise Power
    # SNR_linear = Signal / Noise
    # Noise = Signal / SNR_linear
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_val = (signal_power / snr_linear).item() # Convert to python float
    
    return noise_val

import numpy as np
import torch
from tqdm import tqdm

def generate_precomputed_dataset(scenario_name, n_samples_total, K):
    """
    Generates dataset with PRE-CALCULATED Effective Gains (G).
    
    Returns:
        H_tensor: (N, K, Tx, Sub) [Complex] -> Input for Model
        G_tensor: (N, K, Sub)     [Float]   -> Input for Loss (Effective Gain)
    """
    gen = DeepMIMOGenerator(scenario_name)
    
    # 1. Generate Raw Data (Mixed Difficulty)
    n_half = n_samples_total // 2
    H_easy = gen.generate_dataset(n_half, K, min_corr=0.0, max_corr=0.4, max_gain_ratio=10.0)
    H_hard = gen.generate_dataset(n_half, K, min_corr=0.6, max_corr=0.9, max_gain_ratio=50.0)
    
    H_raw = np.concatenate([H_easy, H_hard], axis=0)
    np.random.shuffle(H_raw)
    
    # Scale for Neural Net stability
    scale_factor = 1e6
    H_scaled = H_raw * scale_factor

    # 2. PRE-COMPUTE 'EFFECTIVE GAINS'
    print("Pre-computing ZF Gains (Fixing Dimensions)...")
    N, K, Tx, Sub = H_scaled.shape
    gains_list = []
    
    for i in range(N):
        H_sample = H_scaled[i] # (K, Tx, Sub)
        sample_gains = []
        for s in range(Sub):
            H_s = H_sample[:, :, s] # (K, Tx)
            
            # --- CORRECTED MATH START ---
            try:
                # 1. Pseudo-Inverse: Shape becomes (Tx, K)
                W = np.linalg.pinv(H_s) 
                
                # 2. Norm of each column (Per User): Shape (K,)
                # We calculate norm along axis 0 (The Tx dimension)
                w_norms = np.linalg.norm(W, axis=0)
                
                # 3. Effective Gain = 1 / ||w||^2
                g = 1.0 / (w_norms**2 + 1e-12)
            except:
                g = np.zeros(K)
            # --- CORRECTED MATH END ---
            
            sample_gains.append(g)
            
        # sample_gains is List of (K,) -> Convert to (Sub, K)
        # Transpose to get (K, Sub)
        gains_list.append(np.array(sample_gains).T) 
        
    G_np = np.array(gains_list) # Result: (N, K, Sub)
    print(f"Gains Shape Verified: {G_np.shape} (Should be {N}, {K}, {Sub})")
    
    # Return as Tensors
    H_tensor = torch.tensor(H_scaled, dtype=torch.complex64)
    G_tensor = torch.tensor(G_np, dtype=torch.float32)
    
    return H_tensor, G_tensor

def solve_water_filling(gains, p_total, noise_val):
    """
    Vectorized Water-Filling Solver.
    Input: (N, K, Sub)
    Output: (N, K, Sub)
    """
    N, K, Sub = gains.shape
    gains_flat = gains.reshape(N, -1)
    P_out = np.zeros_like(gains_flat)
    
    inv_snr = noise_val / (gains_flat + 1e-12)
    
    for i in range(N):
        g = gains_flat[i]
        inv = inv_snr[i]
        sorted_inv = np.sort(inv)
        
        for k in range(len(g), 0, -1):
            wl = (p_total + np.sum(sorted_inv[:k])) / k
            if wl > sorted_inv[k-1]:
                P_out[i] = np.maximum(0, wl - inv)
                break
                
    return P_out.reshape(N, K, Sub)

def compute_scalar_rate(power, gains, noise_val):
    sinr = (power * gains) / (noise_val + 1e-12)
    return np.mean(np.sum(np.log2(1 + sinr), axis=(1, 2)))

class ScalarSumRateLoss(nn.Module):
    def forward(self, power_pred, gains, noise_val):
        """
        Rate = Sum log2(1 + P * Gain / Noise)
        ZERO Beamforming here. Just resource allocation.
        """
        # SINR is now just a simple multiplication
        sinr = (power_pred * gains) / (noise_val + 1e-12)
        rate = torch.log1p(sinr) / np.log(2)
        return -torch.mean(torch.sum(rate, dim=[1, 2]))

# --- MAIN RUNNER ---
def run_no_bf_task(experiment_configs, task_config):
    device = task_config.DEVICE
    print(f"--- Starting Power Allocation (Pre-computed Geometry) ---")
    
    # 1. GET DATA
    K = 4
    # H_tensor: (N, K, Tx, Sub) -> Matches LWM Input Requirement
    # G_tensor: (N, K, Sub)     -> Matches Loss Requirement
    H_tensor, G_tensor = generate_precomputed_dataset(
        task_config.SCENARIO_NAME, n_samples_total=2000, K=K
    )
    
    # Calculate Noise Floor (e.g., 15dB Avg SNR)
    avg_gain = torch.mean(G_tensor).item()
    train_noise = avg_gain / (10**(15.0/10.0))
    
    # 2. SPLIT DATA
    n_total = len(H_tensor)
    idx_bench = int(n_total * 0.8)
    
    X_train_full = H_tensor[:idx_bench]
    G_train_full = G_tensor[:idx_bench]
    
    # Benchmark Data (Keep G on CPU for Numpy Solver)
    X_test = H_tensor[idx_bench:].to(device)
    G_test = G_tensor[idx_bench:].numpy()

    # 3. TRAINING LOOP
    ratios = [0.1, 0.5, 0.8]
    best_models = {}

    # 1. Setup Training Log File
    os.makedirs(task_config.RESULTS_DIR, exist_ok=True)
    train_log_path = os.path.join(task_config.RESULTS_DIR, "training_log.csv")
    with open(train_log_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Ratio", "Model", "Epoch", "Loss", "Train_Rate_bpsHz"])

    for ratio in ratios:
        n_train = int(len(X_train_full) * ratio)
        print(f"\n>>> Training Ratio: {ratio*100}% ({n_train} samples)")
        
        train_ds = TensorDataset(X_train_full[:n_train], G_train_full[:n_train])
        train_dl = DataLoader(train_ds, batch_size=task_config.BATCH_SIZE, shuffle=True)
        
        for config in experiment_configs:
            model = build_model_from_config(config)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = ScalarSumRateLoss()
            
            model.train()
            for epoch in range(task_config.EPOCHS):
                epoch_loss = 0.0
                
                for bx, bg in train_dl:
                    bx = bx.to(device) # Complex H
                    bg = bg.to(device) # Scalar Gains
                    
                    # Forward
                    logits = model(bx)
                    
                    # Output Constraint: Softmax enforces Sum(P) = 1.0
                    p_pred = torch.nn.functional.softmax(logits, dim=1).view(bg.shape)
                    
                    # Loss (Using Scalar Gains Only)
                    loss = criterion(p_pred, bg, train_noise)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # --- LOGGING PER EPOCH ---
                avg_loss = epoch_loss / len(train_dl)
                # Convert Negative Loss back to Rate for readability
                avg_rate = -avg_loss 
                
                # Print to Console (Optional, every 10 epochs to keep it clean)
                if (epoch + 1) % 10 == 0:
                    print(f"   [{config.name}] Epoch {epoch+1:02d}: Loss {avg_loss:.4f} | Rate {avg_rate:.4f} bps/Hz")
                
                # Save to CSV
                with open(train_log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([ratio, config.name, epoch+1, f"{avg_loss:.4f}", f"{avg_rate:.4f}"])
            
            if ratio == 0.8:
                best_models[config.name] = model
            print(f"   Model {config.name} trained.")

    print("\n" + "="*60)
    print(">>> FINAL BENCHMARK: AI (Using H) vs WF (Using Pre-calc G)")
    print("="*60)
    
    results_path = os.path.join(task_config.RESULTS_DIR, "final_benchmark.csv")
    with open(results_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Model", "Test_SNR", "SE_AI", "SE_WF", "Ratio_Pct"])
    
    # Create a DataLoader for the Test Set
    # This prevents sending 3000+ complex matrices to the GPU at once
    test_ds = TensorDataset(X_test)
    test_loader = DataLoader(test_ds, batch_size=task_config.BATCH_SIZE, shuffle=False)
    
    test_snrs = [0, 5, 10, 15, 20, 25, 30]

    for snr in test_snrs:
        # Calculate Noise
        noise_val = avg_gain / (10**(snr/10.0))
        
        # Baseline (WF on Scalar Gains) - CPU is fine for this
        P_wf = solve_water_filling(G_test, p_total=1.0, noise_val=noise_val)
        se_wf = compute_scalar_rate(P_wf, G_test, noise_val)
        
        print(f"\n--- SNR {snr} dB (WF Baseline: {se_wf:.3f}) ---")
        
        for name, model in best_models.items():
            model.eval()
            
            # --- BATCHED INFERENCE START ---
            p_ai_batches = []
            
            with torch.no_grad():
                for (bx_test,) in test_loader:
                    bx_test = bx_test.to(device)
                    
                    # Forward
                    logits = model(bx_test)
                    
                    # Softmax & Move to CPU immediately to free GPU memory
                    # Output: (Batch, K*Sub) or (Batch, K, Sub) depending on model head
                    p_batch = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                    p_ai_batches.append(p_batch)
            
            # Concatenate all batches: (N_total, ...)
            p_ai = np.concatenate(p_ai_batches, axis=0)
            
            # Reshape to match G_test: (N, K, Sub)
            # Ensure G_test.shape is used correctly
            p_ai = p_ai.reshape(G_test.shape)
            # --- BATCHED INFERENCE END ---

            # AI Score
            se_ai = compute_scalar_rate(p_ai, G_test, noise_val)
            
            ratio = (se_ai / se_wf) * 100.0
            
            with open(results_path, 'a', newline='') as f:
                csv.writer(f).writerow([name, snr, f"{se_ai:.4f}", f"{se_wf:.4f}", f"{ratio:.2f}"])
            print(f"   {name:<15}: {ratio:.2f}% Optimal")
    
    print("Done.")