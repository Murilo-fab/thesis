import os
import csv
import time
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

import DeepMIMOv3

from thesis.data_classes import TaskConfig
from thesis.downstream_models import build_model_from_config
from thesis.utils import (create_dataloaders, apply_awgn, get_parameters,
                          get_flops_and_params, get_latency, extract_features)

class BeamPredictionGenerator:
    """
    Handles data generation for Beam Prediction.
    Caches the cleaned channel matrices to allow fast re-labeling for different beam codebooks.
    """
    def __init__(self, scenario_name, scale_factor=1e6):
        self.params = get_parameters(scenario_name)
        self.scale_factor = scale_factor
        
        # Cache for the clean data
        self.clean_chs = None 
        self.clean_locs = None

    def load_raw_data(self):
        """
        Loads and filters data. Only keeps users where LoS != -1.
        """
        print(f"Loading Data for {self.params['scenario']}...")
        
        # 1. Generate Raw Data
        deepmimo_data = DeepMIMOv3.generate_data(self.params)
        
        raw_chs = deepmimo_data[0]['user']['channel']
        raw_los = deepmimo_data[0]['user']['LoS']
        raw_loc = deepmimo_data[0]['user']['location'] # (N, 3)
        
        # 2. Identify Valid Indices
        valid_idxs = np.where(raw_los != -1)[0]
        
        # 3. Filter and Store
        self.clean_chs = raw_chs[valid_idxs]
        self.clean_locs = raw_loc[valid_idxs]
        
        print(f"\tValid Users:    {len(self.clean_chs)}")
        
        return self.clean_chs

    def compute_labels_for_beams(self, n_beams):
        """
        Generates (X, y) for a specific codebook size using cached data.
        """
        if self.clean_chs is None:
            self.load_raw_data()

        # 1. Compute Labels
        beam_indices, valid_mask = self._compute_beam_labels(self.clean_chs, n_beams)
        
        # 2. Final Filter (removing users with 0 power / NaN beams)
        final_chs = self.clean_chs[valid_mask]
        final_labels = beam_indices[valid_mask]
        final_locs = self.clean_locs[valid_mask]
        
        # 3. Format
        if final_chs.ndim == 4:
            final_chs = final_chs.squeeze(axis=1)
            
        X = torch.tensor(final_chs, dtype=torch.complex64) * self.scale_factor
        y = torch.tensor(final_labels, dtype=torch.long)
        
        return X, y, final_locs

    def _compute_beam_labels(self, channels, n_beams):
        """
        Performs exhaustive beam search to find the best beam index for each user.
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

def train_downstream(
    model: nn.Module,
    train_loader, 
    val_loader, 
    task_config: TaskConfig,
) -> tuple[float, nn.Module, dict]:
    """
    Trains a classifier.
    Returns: (current_f1, model, history, total_train_time)
    """
    device = task_config.device
    model.to(device)

    # Initialize Model & Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=task_config.lr)

    crit = nn.CrossEntropyLoss()
    
    # History Container
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    current_f1 = 0.0

    total_train_time = 0.0  # Time spent optimizing (Backprop)

    # Training Loop
    for epoch in range(task_config.epochs):
        # --- Train Phase ---
        t0_train = time.time()
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
        
        if device == 'cuda': torch.cuda.synchronize()
        total_train_time += (time.time() - t0_train)

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

    return current_f1, model, history, total_train_time

def run_beam_selection_task(experiment_configs, task_config: TaskConfig):
    """
    """
    task_name = task_config.task_name
    device = task_config.device
    print(f"Starting Task: {task_name}")
    
    # Setup and data generation
    # Create Timestamped Results Folder
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(task_config.results_dir, task_name, task_config.scenario_name, time_now)
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved to: {results_folder}")

    generator = BeamPredictionGenerator(task_config.scenario_name)
    generator.load_raw_data() 

    # Storage for results
    tsne_results = []
    
    # Log paths
    efficiency_log_file = os.path.join(results_folder, "data_efficiency_results.csv")
    snr_log_file = os.path.join(results_folder, "snr_results.csv")
    resources_log_file = os.path.join(results_folder, "resources_results.csv")

    # Initialize CSV
    with open(efficiency_log_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Num_Beams", "Train_Ratio", "Train_Samples", "Model_Name", "Final_F1", "Training_Time"])
    
    with open(snr_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(["Num_Beams", "SNR_dB", "Model_Name", "F1_Score"])

    with open(resources_log_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Num_Beams", "Model_Name", "MFLOPs", "Params_M", "Encoder_ms", "Head_ms"])

    for config in experiment_configs:
        name = config.name
        
        for n_beams  in task_config.task_complexity:
            print(f"Training model: {name} | Configuration with {n_beams} beams.")
            config.output_size = n_beams

            X_all, y_all, locs = generator.compute_labels_for_beams(n_beams)
            total_samples = len(X_all)

            map_path = os.path.join(results_folder, f"user_map_data_{n_beams}_beams.csv")
            df_map = pd.DataFrame({
                'x': locs[:, 0],
                'y': locs[:, 1],
                'z': locs[:, 2],
                'beam_index': y_all.numpy()
                })
            df_map.to_csv(map_path, index=False)
            
            # 2. Data efficiency
            for ratio in task_config.train_ratios:
                n_train_samples = int(total_samples * ratio)

                # Create DataLoaders
                train_dl, val_dl = create_dataloaders(X_all, y_all, train_ratio=ratio, seed=42)

                # A. Build Fresh Model - This needs some adjustment
                model = build_model_from_config(config)

                # B. Train - This needs some adjustment
                final_f1, model, _, total_train_time = train_downstream(model, train_dl, val_dl, task_config)

                # C. Log
                with open(efficiency_log_file, mode='a', newline='') as f:
                    csv.writer(f).writerow([n_beams, ratio, n_train_samples, name, f"{final_f1:.4f}", f"{total_train_time:.2f}s"])

                print(f"\tRatio {ratio:.4f} | Training Samples: {n_train_samples} | Final F1: {final_f1:.4f}")

            # 3. Resources metrics
            input_sample = X_all[0:1]

            computational_cost = get_flops_and_params(model, input_sample, device)
            latency = get_latency(model, input_sample, device)

            with open(resources_log_file, mode='a', newline='') as f:
                csv.writer(f).writerow([n_beams, name, computational_cost["MFLOPs"], computational_cost["Params_M"],
                                        f"{latency['Encoder_ms']:.4f}", f"{latency['Head_ms']:.4f}"])

            print(f"\tMFLOPs: {computational_cost['MFLOPs']} | Params_M: {computational_cost['Params_M']}",
                  f"| Encoder Latency: {latency['Encoder_ms']:.4f} | Head Latency: {latency['Head_ms']:.4f}")
            
            # 4. Noise robustness

            # Define fixed test set
            _, test_dl_clean = create_dataloaders(X_all, y_all, train_ratio=0.8, val_ratio=0.2, seed=42)

            # Extract the raw tensors to add noise to them manually
            X_test_clean, y_test_fixed = test_dl_clean.dataset.tensors

            for snr in task_config.snr_range:
                # Apply noise dynamically to dataset
                X_test_noisy = apply_awgn(X_test_clean, snr)

                test_ds = torch.utils.data.TensorDataset(X_test_noisy, y_test_fixed)
                test_dl = torch.utils.data.DataLoader(test_ds, batch_size=task_config.batch_size, shuffle=False)

                # List for metrics
                all_preds = []
                all_targets = []

                model.eval()
                with torch.no_grad():
                    for bx, by in test_dl:
                        logits = model(bx.to(device))
                        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        all_targets.extend(by.numpy())

                f1 = f1_score(all_targets, all_preds, average='weighted')

                # Log
                with open(snr_log_file, mode='a', newline='') as f:
                    csv.writer(f).writerow([n_beams, snr, name, f"{f1:.4f}"])

                print(f"\tSNR {snr} | Final F1: {f1:.4f}")

            # 5. Latent space (t-SNE)
            print("\tRunning t-SNE Analysis...")

            vis_ds = torch.utils.data.TensorDataset(X_all, y_all)
            vis_dl = torch.utils.data.DataLoader(vis_ds, batch_size=task_config.batch_size, shuffle=False)

            # A. Extract Latent Features
            features_np, labels_np = extract_features(model, vis_dl, device)

            # B. Compute t-SNE
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
            emb = tsne.fit_transform(features_np)

            # C. Store for CSV
            df_temp = pd.DataFrame({
                'tsne_1': emb[:, 0],
                'tsne_2': emb[:, 1],
                'label': labels_np,
                'model_name': name,
                'n_beams': n_beams
                })
            tsne_results.append(df_temp)
            print("\tt-SNE Analysis Complete")

    # Save CSV
    tsne_csv_path = os.path.join(results_folder, "tsne_comparison_data.csv")
    if tsne_results:
        pd.concat(tsne_results, ignore_index=True).to_csv(tsne_csv_path, index=False)
        print(f"   t-SNE data saved to: {tsne_csv_path}")

                