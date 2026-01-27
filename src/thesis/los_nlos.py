import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import DeepMIMOv3 

from thesis.utils import get_parameters, create_dataloaders

from dataclasses import dataclass, asdict

@dataclass
class TaskConfig:
    # Identity
    TASK_NAME: str = "LoS_NLoS_Classification"

    # Single Scenario Name
    SCENARIO_NAME: str = "city_6_miami" 
    
    # Ratios to test (e.g., 1%, 10%, 50%, 100% of the training pool)
    TRAIN_RATIOS: list = (.001, .01, .05, .1, .25, .5, .8,)
    
    # SNR levels to test robustness
    SNR_RANGE: list = (-5, 0, 5, 10, 15, 20)
    
    BATCH_SIZE: int = 32
    EPOCHS: int = 15
    LR: float = 1e-3
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    RESULTS_DIR: str = f"./results/"

    def save_to_json(self, save_dir):
        """Saves this config state to the results folder for reproducibility."""
        path = os.path.join(save_dir, "config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"Config saved to {path}")


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
            return chs, labels
        
        except Exception as e:
            raise ValueError(f"DeepMIMO Generation Failed: {e}")

import time

def train_downstream(
    model: nn.Module,
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    task_config: TaskConfig,
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

import csv
from datetime import datetime
from thesis.utils import build_model_from_config
import pandas as pd

def run_los_nlos_task(experiment_configs: list, task_config: TaskConfig):
    """
    Runs a complete analysis on a single DeepMIMO scenario:
    1. Data Efficiency (Ratio Sweep) -> Logs to CSV & Caches Best Models
    2. Noise Robustness (SNR Sweep) -> Logs to CSV
    3. Performance Visualization -> Saves performance_metrics.png
    4. Latent Space Visualization (t-SNE) -> Saves latent_space_comparison.png
    """
    device = task_config.DEVICE
    print(f"--- Starting Task: {task_config.TASK_NAME} ---")

    # 1. SETUP & DATA GENERATION
    # Create Timestamped Results Folder
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(task_config.RESULTS_DIR, task_config.TASK_NAME, time_now)
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved to: {results_folder}")
    
    # Generate Data
    print(f"\n[1/4] Generating Dataset for {task_config.SCENARIO_NAME}...")
    generator = DeepMIMOGenerator(task_config.SCENARIO_NAME)
    X_all, y_all = generator.generate_dataset()
    total_samples = len(X_all)
    
    # Storage for results
    ratio_results = {cfg.name: [] for cfg in experiment_configs}
    snr_results = {cfg.name: [] for cfg in experiment_configs}
    trained_models_cache = {} # Cache models trained on the largest data ratio
    
    # 2. DATA EFFICIENCY SWEEP (Ratio Loop)
    print("\n[2/4] Starting Training Ratio Sweep...")
    efficiency_log_file = os.path.join(results_folder, "data_efficiency_results.csv")

    # Initialize CSV
    with open(efficiency_log_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Train_Ratio", "Train_Samples", "Model_Name", "Final_Val_F1"])

    for ratio in task_config.TRAIN_RATIOS:
        n_train_samples = int(total_samples * ratio)
        print(f"\n   > Ratio {ratio:.4f} | Training Samples: {n_train_samples}")

        # Create DataLoaders
        train_dl, val_dl = create_dataloaders(X_all, y_all, train_ratio=ratio, seed=42)

        for config in experiment_configs:
            # A. Build Fresh Model
            model = build_model_from_config(config)

            # B. Train (Returns result from LAST epoch
            run_name = f"{config.name}_{ratio}"
            final_f1, head_model, _ = train_downstream(
                model, train_dl, val_dl, task_config, run_name, results_folder
            )

            # C. Log & Store
            ratio_results[config.name].append(final_f1)
            
            with open(efficiency_log_file, mode='a', newline='') as f:
                csv.writer(f).writerow([ratio, n_train_samples, config.name, f"{final_f1:.4f}"])

            # D. Cache Model (only if this is the max ratio)
            if ratio == task_config.TRAIN_RATIOS[-1]:
                trained_models_cache[config.name] = head_model
        
            print(f"     Model: {config.name:<15} | Final F1: {final_f1:.4f}")

    # 3. NOISE ROBUSTNESS SWEEP (SNR Loop)
    print("\n[3/4] Starting SNR Sweep (using models trained on full data)...")
    snr_log_file = os.path.join(results_folder, "snr_results.csv")

    # Initialize CSV
    with open(snr_log_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["SNR_dB", "Model_Name", "F1_Score"])

    # Define Fixed Test Set
    _, test_dl_clean = create_dataloaders(X_all, y_all, train_ratio=0.8, val_ratio=0.2, seed=42)

    # Extract the raw tensors so we can add noise to them manually
    X_test_clean, y_test_fixed = test_dl_clean.dataset.tensors

    for snr in task_config.SNR_RANGE:
        # Apply Noise dynamically to dataset
        X_test_noisy = apply_awgn(X_test_clean, snr)

        test_ds = torch.utils.data.TensorDataset(X_test_noisy, y_test_fixed)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=task_config.BATCH_SIZE, shuffle=False)

        for config in experiment_configs:
            name = config.name
            if name not in trained_models_cache: continue

            # Evaluate Cached Model
            head = trained_models_cache[name]
            head.eval()
            
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for bx, by in test_dl:
                    logits = head(bx.to(device))
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_targets.extend(by.numpy())
            
            f1 = f1_score(all_targets, all_preds, average='weighted')
            snr_results[name].append(f1)

            # Log
            with open(snr_log_file, mode='a', newline='') as f:
                csv.writer(f).writerow([snr, name, f"{f1:.4f}"])
            
        print(f"   > SNR {snr} dB completed.")

    # 4. VISUALIZATION A: PERFORMANCE METRICS
    print("\n[4/4] Generating Plots...")

    # Create Figure 1: Performance Curves
    plt.figure(figsize=(12, 5))

    # Subplot 1: Ratio vs F1
    plt.subplot(1, 2, 1)
    for name, scores in ratio_results.items():
        if len(scores) == len(task_config.TRAIN_RATIOS):
            plt.plot(task_config.TRAIN_RATIOS, scores, 'o-', linewidth=2, label=name)
    plt.xlabel("Number of training samples (%)")
    plt.ylabel("F1 Score")
    plt.title("Data Efficiency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: SNR vs F1
    plt.subplot(1, 2, 2)
    for name, scores in snr_results.items():
        if len(scores) == len(task_config.SNR_RANGE):
            plt.plot(task_config.SNR_RANGE, scores, 's--', linewidth=2, label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("F1 Score")
    plt.title("Noise Robustness")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path_metrics = os.path.join(results_folder, "performance_metrics.png")
    plt.tight_layout()
    plt.savefig(save_path_metrics, dpi=300)
    plt.close() # Close to free memory
    print(f"   Performance metrics saved to: {save_path_metrics}")

    # 5. VISUALIZATION B: LATENT SPACE (t-SNE)
    print("   Running t-SNE Analysis...")

    # 1. Identify Models to Plot (Filter for LWM, AE, Raw)
    keywords = ["LWM", "AE", "Raw"] 
    targets = [name for name in trained_models_cache.keys() if any(k in name for k in keywords)]

    if not targets:
        print("   No matching models found for t-SNE.")
        return
    
    # 2. Select Data Subset (Consistent across all models)
    N_vis = 1000
    if len(X_all) > N_vis:
        np.random.seed(42) 
        idx_vis = np.random.choice(len(X_all), N_vis, replace=False)
        X_vis = X_all[idx_vis]
        y_vis = y_all[idx_vis]
    else:
        X_vis, y_vis = X_all, y_all

    vis_ds = torch.utils.data.TensorDataset(X_vis, y_vis)
    vis_dl = torch.utils.data.DataLoader(vis_ds, batch_size=task_config.BATCH_SIZE, shuffle=False)

    # 3. Process Each Model
    tsne_results = []
    num_plots = len(targets)

    # Create Figure 2: t-SNE Comparison
    plt.figure(figsize=(5 * num_plots, 5))

    for i, name in enumerate(targets):
        print(f"     Processing t-SNE for {name}...")
        model = trained_models_cache[name]

        # A. Extract Latent Features
        features_tensor, labels_tensor = extract_latent_features(model, vis_dl, device)
        features_np = features_tensor.numpy()
        labels_np = labels_tensor.numpy()

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

        # D. Plot Subplot
        plt.subplot(1, num_plots, i+1)
        scatter = plt.scatter(
            emb[:, 0], emb[:, 1], 
            c=labels_np, 
            cmap='coolwarm', 
            s=15, alpha=0.6
        )
        plt.title(name)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True, alpha=0.3)

        # Legend (only on last plot)
        if i == num_plots - 1:
             handles, _ = scatter.legend_elements(prop="colors")
             plt.legend(handles, ["NLoS", "LoS"], title="Class", loc="upper right")

    # Save CSV
    tsne_csv_path = os.path.join(results_folder, "tsne_comparison_data.csv")
    if tsne_results:
        pd.concat(tsne_results, ignore_index=True).to_csv(tsne_csv_path, index=False)
        print(f"   t-SNE data saved to: {tsne_csv_path}")

    # Save Figure
    tsne_plot_path = os.path.join(results_folder, "latent_space_comparison.png")
    plt.tight_layout()
    plt.savefig(tsne_plot_path, dpi=300)
    plt.close()
    print(f"   t-SNE plot saved to: {tsne_plot_path}")

    print("\n--- Analysis Complete ---")

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

def extract_latent_features(model, dataloader, device):
    """
    Runs inference but stops before the classification head to get latent features.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            
            # 1. Handle LWMWrapper
            if hasattr(model, 'lwm_model'): 
                # Replicate the forward pass logic manually up to the head
                # (Batch, 32, 32) -> Tokens
                if bx.ndim == 4: # Multi-user case logic if needed
                    B, K, M, N = bx.shape
                    tokens = model.tokenizer(bx.view(B*K, M, N))
                else:
                    tokens = model.tokenizer(bx)
                
                # LWM Forward
                embeddings, _ = model.lwm_model(tokens)
                
                # Extraction Mode (cls vs channel_emb)
                if model.mode == "cls":
                    feats = embeddings[:, 0, :]
                elif model.mode == "channel_emb":
                    feats = embeddings[:, 1:, :].flatten(start_dim=1)
                elif model.mode == "mean_pooled":
                     feats = torch.mean(embeddings, dim=1)
                
            # 2. Handle CSIAEWrapper
            elif hasattr(model, 'ae_model'):
                # (Batch, 32, 32) -> Latent
                # We assume the AE model has a 'forward' that returns (z, recon)
                # OR we run encoder manualy if it's a wrapper
                
                # Robust approach: Run the AE encoder part
                # Check if it's the class with 'encoder' and 'fc_enc'
                ae = model.ae_model
                x_real = torch.stack([bx.real, bx.imag], dim=1).float()
                x_norm = ae.input_norm(x_real)
                enc_out = ae.encoder(x_norm)
                feats = ae.fc_enc(enc_out)
                
            # 3. Handle LinearWrapper / Raw
            else:
                # For Raw Linear, the "feature" is just the flattened input
                feats = torch.view_as_real(bx).flatten(start_dim=1)
                
            features_list.append(feats.cpu())
            labels_list.append(by)
            
    return torch.cat(features_list), torch.cat(labels_list)