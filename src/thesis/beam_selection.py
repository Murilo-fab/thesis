import os
import csv
from tqdm import tqdm
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import accuracy_score

import DeepMIMOv3

from thesis.utils import get_parameters
from thesis.utils import build_model_from_config, create_dataloaders

class BeamPredictionGenerator:
    def __init__(self, scenario_name, scale_factor=1e6):
        self.params = get_parameters(scenario_name)
        self.scale_factor = scale_factor
        
        # Cache for the clean data
        self.clean_chs = None 
        self.clean_locs = None # Useful if you plot coverage later

    def load_raw_data(self):
        """
        Loads and filters data. 
        Only keeps users where LoS != -1.
        """
        print(f"Loading Data for {self.params['scenario']}...")
        
        # 1. Generate Raw Data
        deepmimo_data = DeepMIMOv3.generate_data(self.params)
        
        raw_chs = deepmimo_data[0]['user']['channel']
        raw_los = deepmimo_data[0]['user']['LoS']
        raw_loc = deepmimo_data[0]['user']['location'] # (N, 3)
        
        # 2. Identify Valid Indices Immediately
        # We only care about users that exist in the simulation
        valid_idxs = np.where(raw_los != -1)[0]
        
        # 3. Filter and Store
        # We reduce the dataset size by ~30-50% here instantly
        self.clean_chs = raw_chs[valid_idxs]
        self.clean_locs = raw_loc[valid_idxs]
        
        print(f"   Original Users: {len(raw_los)}")
        print(f"   Valid Users:    {len(self.clean_chs)}")
        
        return self.clean_chs

    def compute_labels_for_beams(self, n_beams):
        """
        Generates X, y for a specific number of beams using the pre-filtered data.
        """
        if self.clean_chs is None:
            self.load_raw_data()

        # 1. Compute Labels
        # We pass the clean channels. No need to pass 'los' anymore since we know they are valid.
        beam_indices, valid_mask = self._compute_beam_labels(self.clean_chs, n_beams)
        
        # 2. Final Filter (removing users with 0 power / NaN beams)
        # Even if LoS != -1, beamforming might technically fail (yield 0)
        final_chs = self.clean_chs[valid_mask]
        final_labels = beam_indices[valid_mask]
        
        # 3. Format
        if final_chs.ndim == 4:
            final_chs = final_chs.squeeze(axis=1)
            
        X = torch.tensor(final_chs, dtype=torch.complex64) * self.scale_factor
        y = torch.tensor(final_labels, dtype=torch.long)
        
        return X, y

    def _compute_beam_labels(self, channels, n_beams):
        """
        Simplified beam search. No 'if los == -1' check needed.
        """
        n_users = len(channels)
        
        # --- Codebook Setup (Same as before) ---
        fov = 180
        beam_angles = np.around(np.arange(-fov/2, fov/2+.1, fov/(n_beams-1)), 2)
        bs_ant_params = self.params['bs_antenna']
        kd_val = 2 * np.pi * bs_ant_params['spacing']
        
        codebook_list = []
        for azi in beam_angles:
            phi_rad = azi * np.pi / 180
            sv = self._steering_vec(bs_ant_params['shape'], phi_rad, 0, kd_val)
            codebook_list.append(sv.squeeze())
        F = np.array(codebook_list) # (N_Beams, N_Tx)
        
        # --- Fast Beam Sweeping ---
        best_beams = np.zeros(n_users, dtype=int)
        valid_mask = np.zeros(n_users, dtype=bool)
        
        # Vectorized batch processing is possible here, but loop is fine for readability
        for i in tqdm(range(n_users), desc=f"   Beam Search ({n_beams})"):
            
            # 1. Prepare Channel
            h_user = channels[i].squeeze() 
            if h_user.ndim == 2:
                h_user = np.mean(h_user, axis=1) 
            
            # 2. Beamform (Linear Power)
            bf_power = np.abs(F @ h_user) ** 2
            
            # 3. Select Best
            best_idx = np.argmax(bf_power)
            
            # Safety check for numerical zeros
            if bf_power[best_idx] > 1e-9: 
                best_beams[i] = best_idx
                valid_mask[i] = True
                
        return best_beams, valid_mask
    
    @staticmethod
    def _steering_vec(array, phi, theta, kd):
        idxs = DeepMIMOv3.ant_indices(array)
        resp = DeepMIMOv3.array_response(idxs, phi, theta + np.pi/2, kd)
        return resp / np.linalg.norm(resp)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from sklearn.metrics import f1_score

def train_downstream(
    model: nn.Module,
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    task_config,
    run_name: str,
    results_folder: str
) -> tuple[float, nn.Module, dict]:
    """
    Trains a classifier and returns the model.
    """
    device = task_config.DEVICE
    model.to(device)

    # 1. Setup Log
    os.makedirs(results_folder, exist_ok=True)
    log_file = os.path.join(results_folder, f"training_log_{run_name}.csv")

    headers = ["Epoch", "Train Loss", "Validation Loss", "Val F1", "Learning Rate", "Time(s)"]

    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    # 2. Initialize Model & Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=task_config.LR)

    crit = nn.CrossEntropyLoss()
    
    # History Container
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    start_time = time.time()
    current_f1 = 0.0

    # 3. Training Loop
    for epoch in range(task_config.EPOCHS):
        # --- Train Phase ---
        model.train()
        train_loss_sum = 0
        train_batches = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = crit(logits, by)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
            
        avg_train_loss = train_loss_sum / train_batches
        history['train_loss'].append(avg_train_loss)
            
        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0
        val_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                
                logits = model(bx)
                loss = crit(logits, by)
                
                val_loss_sum += loss.item()
                val_batches += 1
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(by.cpu().numpy())
        
        # Calculate Epoch Metrics
        avg_val_loss = val_loss_sum / val_batches
        current_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(current_f1)

        # Timing
        elapsed_time = time.time() - start_time
        
        # --- Logging to CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{current_f1:.6f}",
                f"{elapsed_time:.2f}"
            ])

    return current_f1, model, history

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class TaskConfig:
    TASK_NAME: str = "Beam_Selection"
    SCENARIO_NAME: str = "city_6_miami"
    RESULTS_DIR: str = "./results"
    
    # Sweep Parameters
    TRAIN_RATIOS: list = (0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.80)
    SNR_RANGE: list = (-5, 0, 5, 10, 15, 20)
    
    # Training Hyperparameters
    BATCH_SIZE: int = 128
    EPOCHS: int = 50
    LR: float = 1e-3
    DEVICE: str = "cuda"


def run_beam_selection_task(experiment_configs, task_config: TaskConfig):
    """
    Runs a hierarchical sweep for Beam Prediction:
    Outer Loop: Codebook Size (16, 32, 64, 128, 256 beams)
    Inner Loop: Data Efficiency (10% -> 100% data)
    Final Step: SNR Robustness sweep for the best models of each beam size.
    """
    device = task_config.DEVICE
    print(f"--- Starting Task: {task_config.TASK_NAME} (Beam Selection) ---")
    
    # ====================================================
    # 1. SETUP & INITIALIZATION
    # ====================================================
    # Create Results Directory
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(task_config.RESULTS_DIR, task_config.TASK_NAME, time_now)
    os.makedirs(results_folder, exist_ok=True)
    
    # Initialize Generator (Loads raw data ONCE)
    # We start with a dummy beam count; it will be updated in the loop.
    print(f"\n[1/5] Initializing Generator for {task_config.SCENARIO_NAME}...")
    generator = BeamPredictionGenerator(task_config.SCENARIO_NAME)
    generator.load_raw_data() # Optimally filters out invalid (LoS=-1) users immediately

    # Define Sweep Parameters
    BEAM_COUNTS = [16, 32, 64, 128, 256] 
    # Use ratios from task_config (e.g., [0.1, ... 1.0])
    
    # Result Containers for Plotting
    # structure: results[n_beams][model_name] = [scores...]
    efficiency_results = {} 
    snr_results = {}
    
    # CSV Headers
    eff_log_file = os.path.join(results_folder, "beam_efficiency_results.csv")
    snr_log_file = os.path.join(results_folder, "beam_snr_results.csv")
    
    with open(eff_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Beams", "Train_Ratio", "Train_Samples", "Model_Name", "Weighted_F1"])
    with open(snr_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Beams", "SNR_dB", "Model_Name", "Accuracy"])

    # ====================================================
    # 2. OUTER LOOP: BEAMBOOK SIZE (Complexity)
    # ====================================================
    for n_beams in BEAM_COUNTS:
        print(f"\n" + "="*50)
        print(f" CONFIGURATION: {n_beams} BEAMS")
        print(f"="*50)
        
        efficiency_results[n_beams] = {cfg.name: [] for cfg in experiment_configs}
        snr_results[n_beams] = {cfg.name: [] for cfg in experiment_configs}
        
        # A. Update Labels
        # Re-run beam search on the cached clean channels
        print(f"   > Computing labels for {n_beams} beams...")
        X, y = generator.compute_labels_for_beams(n_beams)
        total_samples = len(X)
        print(f"   > Dataset ready: {total_samples} samples.")

        # B. Update Model Configs
        # Crucial: Resize the classifier output head to n_beams
        for cfg in experiment_configs:
            cfg.head_args['num_classes'] = n_beams

        # C. Define Fixed Test Set for SNR (for this beam config)
        # We split once per beam-config to ensure consistent SNR testing
        _, test_dl_clean = create_dataloaders(X, y, train_ratio=0.8, val_ratio=0.2, seed=42)
        X_test_clean, y_test_fixed = test_dl_clean.dataset.tensors
        
        # Cache for models trained on 100% data (for SNR sweep)
        current_beam_models = {}

        # ====================================================
        # 3. INNER LOOP: DATA EFFICIENCY (Ratios)
        # ====================================================
        print(f"   > Starting Ratio Sweep...")
        for ratio in task_config.TRAIN_RATIOS:
            n_train_samples = int(total_samples * ratio)
            train_dl, val_dl = create_dataloaders(X, y, train_ratio=ratio, seed=42)
            
            for config in experiment_configs:
                # 1. Build Model (Head size is now n_beams)
                model = build_model_from_config(config)
                
                # 2. Train
                run_name = f"B{n_beams}_{config.name}_R{ratio}"
                score, trained_model, _ = train_downstream(
                    model, train_dl, val_dl, task_config, run_name, results_folder
                )
                
                # 3. Log
                efficiency_results[n_beams][config.name].append(score)
                with open(eff_log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([n_beams, ratio, n_train_samples, config.name, f"{score:.4f}"])
                
                # 4. Cache if Max Ratio
                if ratio == task_config.TRAIN_RATIOS[-1]:
                    current_beam_models[config.name] = trained_model
            
            # Print brief status for the first model in list
            ref_model = experiment_configs[0].name
            print(f"     Ratio {ratio:.2f}: {ref_model} = {efficiency_results[n_beams][ref_model][-1]:.4f}")

        # ====================================================
        # 4. SNR SWEEP (For this Beam Count)
        # ====================================================
        print(f"   > Starting SNR Sweep (on {n_beams}-beam models)...")
        
        for snr in task_config.SNR_RANGE:
            # Add noise to the FIXED test set
            X_test_noisy = apply_awgn(X_test_clean, snr)
            
            test_ds = torch.utils.data.TensorDataset(X_test_noisy, y_test_fixed)
            test_dl = torch.utils.data.DataLoader(test_ds, batch_size=task_config.BATCH_SIZE, shuffle=False)
            
            for config in experiment_configs:
                name = config.name
                if name not in current_beam_models: continue
                
                model = current_beam_models[name]
                model.eval()
                
                # Inference
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for bx, by in test_dl:
                        logits = model(bx.to(device))
                        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        all_targets.extend(by.numpy())
                
                # Metric: Accuracy is preferred for SNR sweeps (easier to interpret)
                acc = accuracy_score(all_targets, all_preds)
                snr_results[n_beams][name].append(acc)
                
                with open(snr_log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([n_beams, snr, name, f"{acc:.4f}"])

    # ====================================================
    # 5. VISUALIZATION
    # ====================================================
    print("\n[5/5] Generating Plots...")
    
    # We generate 2 Plot Files:
    # 1. Complexity vs Efficiency (How hard is 256 beams vs 16?)
    # 2. SNR Robustness (How fast does 256 beams fail vs 16?)
    
    # --- Plot 1: Efficiency Curves for different Beam Counts (Target Model Only) ---
    target_model = experiment_configs[0].name # Usually "LWM" or "AE"
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(BEAM_COUNTS)))
    
    for i, n_beams in enumerate(BEAM_COUNTS):
        if target_model in efficiency_results[n_beams]:
            scores = efficiency_results[n_beams][target_model]
            plt.plot(task_config.TRAIN_RATIOS, scores, 'o-', color=colors[i], label=f"{n_beams} Beams")
            
    plt.title(f"Impact of Codebook Size on Efficiency ({target_model})")
    plt.xlabel("Training Ratio")
    plt.ylabel("Weighted F1 Score")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Codebook Size")
    plt.savefig(os.path.join(results_folder, "beam_complexity_efficiency.png"), dpi=300)
    
    # --- Plot 2: SNR Curves for different Beam Counts ---
    plt.figure(figsize=(10, 6))
    
    for i, n_beams in enumerate(BEAM_COUNTS):
        if target_model in snr_results[n_beams]:
            scores = snr_results[n_beams][target_model]
            plt.plot(task_config.SNR_RANGE, scores, 's--', color=colors[i], label=f"{n_beams} Beams")

    plt.title(f"Impact of Codebook Size on Robustness ({target_model})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Codebook Size")
    plt.savefig(os.path.join(results_folder, "beam_complexity_snr.png"), dpi=300)

    print("\nDone. All results saved.")
    # plt.show()

def apply_awgn(x_complex, snr_db):
    """
    Applies Additive White Gaussian Noise to a batch of complex channels.
    """
    # x_complex: (N, 32, 32)
    # Calculate signal power per sample
    sig_power = torch.mean(torch.abs(x_complex)**2, dim=(1,2), keepdim=True)
    
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_std = torch.sqrt(noise_power / 2)
    
    # Noise must match device of input, but usually applied on CPU before norm
    noise = torch.randn_like(x_complex) * noise_std + 1j * torch.randn_like(x_complex) * noise_std
    return x_complex + noise