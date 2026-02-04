import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import DeepMIMOv3 

import csv
from datetime import datetime
import pandas as pd

from thesis.utils import (get_parameters, create_dataloaders,
                          apply_awgn, extract_features,
                          get_flops_and_params, get_latency)

from thesis.downstream_models import build_model_from_config
from thesis.data_classes import TaskConfig


class DeepMIMOGenerator:
    """
    Handles the configuration and generation of wireless channel datasets using the DeepMIMOv3 engine.
    
    This wrapper simplifies the process of loading a specific scenario, generating raw channel
    data, filtering out invalid users, and converting the result into PyTorch tensors
    ready for training or evaluation.
    """
    def __init__(self, scenario_name, scale_factor=1e6):
        """
        Initializes the generator by loading the parameters for the specified scenario.

        Args:
            scenario_name (str): The name of the DeepMIMO scenario folder (e.g., 'city_18_denver').
                                 This must match a key in the parameter configuration dictionary.
            scale_factor (float): Fixed multiplier applied to raw channels.
        """
        self.params = get_parameters(scenario_name)
        self.scale_factor = scale_factor

    def generate_dataset(self):
        """
        Generates the complex channel dataset and corresponding Line-of-Sight (LoS) labels.

        This method performs the following steps:
        1. Calls the DeepMIMO engine to generate raw channel data based on `self.params`.
        2. Identifies valid users (those with a valid LoS path).
        3. Extracts the channel matrix for valid users.
        4. Removes the singleton Receiver Antenna dimension (since UE antenna is 1).
        5. Converts the data to PyTorch Complex64 tensors.

        Returns:
            tuple:
                - chs (torch.Tensor): The complex channel matrices. 
                  Shape: (N_Samples, N_Tx_Antennas, N_Subcarriers) 
                  Example: (2000, 32, 32)
                - labels (torch.Tensor): The LoS status labels (0 = NLoS, 1 = LoS).
                  Shape: (N_Samples,)
        
        Raises:
            ValueError: If the DeepMIMO engine fails to generate data (e.g., missing files).
        """
        try:
            # 1. Generate raw data using the DeepMIMO engine
            deepmimo_data = DeepMIMOv3.generate_data(self.params)

            # 2. Filter valid users
            # LoS value of -1 indicates the user is invalid or not active
            los = deepmimo_data[0]['user']['LoS']
            valid = np.where(los != -1)[0]
            
            # 3. Extract channels for valid users only
            # Raw Shape: (N_All_Users, 1_Rx, N_Tx, N_Subcarriers)
            raw_chs = deepmimo_data[0]['user']['channel'][valid]
            raw_locs = deepmimo_data[0]['user']['location'][valid]

            # 4. Squeeze RX dim
            # The receiver usually has 1 antenna, so we remove that dimension.
            # Shape becomes: (N_Valid, N_Tx, N_Subcarriers)
            if raw_chs.ndim == 4:
                raw_chs = raw_chs.squeeze(axis=1)

            # 5. Convert to PyTorch Tensors
            # Channels use Complex64 (32-bit float real + 32-bit float imag)
            chs = torch.tensor(raw_chs, dtype=torch.complex64) * self.scale_factor
            labels = torch.tensor(los[valid], dtype=torch.long)
            
            print(f"Generated {len(chs)} samples. Final Shape: {chs.shape}")
            return chs, labels, raw_locs
        
        except Exception as e:
            raise ValueError(f"DeepMIMO Generation Failed: {e}")

def train_downstream(
    model: nn.Module,
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    task_config: TaskConfig
) -> tuple[float, nn.Module, dict]:
    """
    Trains a classifier and returns the current_f1, model, history, total_train_time.
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

def run_los_nlos_task(experiment_configs: list, task_config: TaskConfig):
    """
    Runs a complete analysis on a single DeepMIMO scenario:
    1. Data Efficiency (Ratio Sweep)
    2. Noise Robustness (SNR Sweep)
    3. Latent Space Visualization (t-SNE)
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
    
    # Generate Data
    print(f"\nGenerating dataset for {task_config.scenario_name}...")
    generator = DeepMIMOGenerator(task_config.scenario_name)
    X_all, y_all, locs = generator.generate_dataset()
    total_samples = len(X_all)

    map_path = os.path.join(results_folder, "user_map_los_nlos.csv")
    df_map = pd.DataFrame({
        'x': locs[:, 0],
        'y': locs[:, 1],
        'z': locs[:, 2],
        'label': y_all.numpy()
    })
    df_map.to_csv(map_path, index=False)
    
    # Storage for results
    ratio_results = {cfg.name: [] for cfg in experiment_configs}
    snr_results = {cfg.name: [] for cfg in experiment_configs}
    tsne_results = []
    
    # Log paths
    efficiency_log_file = os.path.join(results_folder, "data_efficiency_results.csv")
    snr_log_file = os.path.join(results_folder, "snr_results.csv")
    resources_log_file = os.path.join(results_folder, "resources_results.csv")

    # Initialize CSV
    with open(efficiency_log_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Train_Ratio", "Train_Samples", "Model_Name", "Final_F1", "Training_Time"])

    with open(snr_log_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["SNR_dB", "Model_Name", "F1_Score"])

    with open(resources_log_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Model_Name", "MFLOPs", "Params_M", "Encoder_ms", "Head_ms"])

    for config in experiment_configs:
        name = config.name
        print(f"Training model: {name}")
        # 2. Data efficiency
        for ratio in task_config.train_ratios:
            n_train_samples = int(total_samples * ratio)
        
            # Create DataLoaders
            train_dl, val_dl = create_dataloaders(X_all, y_all, train_ratio=ratio, seed=42)

            # A. Build Fresh Model - This needs some adjustment
            model = build_model_from_config(config)

            # B. Train - This needs some adjustment
            final_f1, model, _, total_train_time = train_downstream(model, train_dl, val_dl, task_config)

            # C. Log & Store
            ratio_results[name].append(final_f1)
            
            with open(efficiency_log_file, mode='a', newline='') as f:
                csv.writer(f).writerow([ratio, n_train_samples, name, f"{final_f1:.4f}", f"{total_train_time:.2f}s"])

            print(f"\tRatio {ratio:.4f} | Training Samples: {n_train_samples} | Final F1: {final_f1:.4f}")

        # 3. Resources metrics
        input_sample = X_all[0:1]

        computational_cost = get_flops_and_params(model, input_sample, device)
        latency = get_latency(model, input_sample, device)

        with open(resources_log_file, mode='a', newline='') as f:
            csv.writer(f).writerow([name, computational_cost["MFLOPs"], computational_cost["Params_M"],
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
            snr_results[name].append(f1)

            # Log
            with open(snr_log_file, mode='a', newline='') as f:
                csv.writer(f).writerow([snr, name, f"{f1:.4f}"])

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
            'model_name': name
        })
        tsne_results.append(df_temp)
        print("\tt-SNE Analysis Complete")

    # Save CSV
    tsne_csv_path = os.path.join(results_folder, "tsne_comparison_data.csv")
    if tsne_results:
        pd.concat(tsne_results, ignore_index=True).to_csv(tsne_csv_path, index=False)
        print(f"   t-SNE data saved to: {tsne_csv_path}")


    