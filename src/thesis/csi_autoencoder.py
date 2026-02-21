"""
CSI Autoencoder Pre-training Script.

This script trains a Convolutional Autoencoder (CAE) to compress and denoise 
Channel State Information (CSI) matrices. 

Key Features:
1. Multi-City Training: Aggregates data from multiple DeepMIMO scenarios to ensure generalization.
2. Physics-Aware Denoising: Uses a fixed noise floor (based on system reference power) 
   during training to teach the AE to handle path loss and thermal noise correctly.
3. NMSE Loss: Optimizes for Normalized Mean Squared Error.

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

# --- 1. Standard Library Imports ---
import os
import csv
import time
from datetime import datetime

# --- 2. Third-Party Imports ---
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from thesis.data_classes import AutoEncoderConfig

# --- 3. Local Imports ---
import DeepMIMOv3
from thesis.utils import get_parameters, apply_awgn

# -----------------------------------------------------------------------------
# MODEL ARCHITECTURE
# -----------------------------------------------------------------------------

class RefineBlock(nn.Module):
    """
    A Residual Block that refines features without changing spatial dimensions.
    Used in the Decoder to reconstruct fine details.
    
    Shape: (B, C, H, W) -> (B, C, H, W)
    """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.net(x)
        return self.relu(out + residual)

class CSIAutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for CSI Compression and Denoising.
    
    Input: Complex CSI (B, Tx, Subcarriers) -> Treated as 2-channel image (Real, Imag).
    Latent: Compressed vector (B, Latent_Dim).
    Output: Reconstructed Complex CSI.
    """
    def __init__(self, latent_dim=64, mode="inference"):
        super().__init__()
        self.mode = mode
        
        # --- Encoder ---
        # Input: (B, 2, 32, 32)
        self.input_norm = nn.BatchNorm2d(2)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (B, 64, 16, 16)
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (B, 128, 8, 8)
            
            nn.Flatten()     # -> (B, 128*8*8) = (B, 8192)
        )
        
        self.flat_dim = 128 * 8 * 8
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        
        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        
        self.decoder_initial = nn.Sequential(
            # Unflatten happens in forward
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.refine1 = nn.Sequential(
            RefineBlock(64),
            RefineBlock(64)
        )
        
        self.decoder_final = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 2, 32, 32)
        )

    def load_weights(self, path, device="cpu"):
        """Safely loads model weights from a .pth file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")
            
        try:
            state_dict = torch.load(path, map_location=device)
            # Handle DataParallel keys
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
            self.load_state_dict(state_dict)
            self.to(device)
            self.eval()
        except RuntimeError as e:
            print(f"Error loading weights: {e}. Check latent_dim config.")
            raise e

    def forward(self, x):
        """
        Args:
            x (Tensor): Complex input [B, Tx, SC]
        Returns:
            If mode='inference': z (Latent Vector)
            If mode='train': (z, x_recon)
        """
        # 1. Preprocess: Complex -> 2-Channel Real
        x_real = torch.stack([x.real, x.imag], dim=1).float() # (B, 2, Tx, SC)
        x_norm = self.input_norm(x_real)
        
        # 2. Encode
        feat = self.encoder(x_norm)
        z = self.fc_enc(feat)
        
        if self.mode == "inference":
            return z
        
        # 3. Decode
        x_recon = self.fc_dec(z).view(-1, 128, 8, 8)
        x_recon = self.decoder_initial(x_recon)
        x_recon = self.refine1(x_recon)
        x_recon = self.decoder_final(x_recon)
        
        # 4. Postprocess: 2-Channel Real -> Complex
        return z, torch.complex(x_recon[:,0], x_recon[:,1])

# -----------------------------------------------------------------------------
# DATA UTILITIES
# -----------------------------------------------------------------------------

class MultiCityGenerator:
    """
    Aggregates data from multiple DeepMIMO scenarios into a single dataset.
    This creates a diverse training set for the Autoencoder.
    """
    def __init__(self, scenario_list: list, scale_factor: float = 1e6):
        self.scenario_list = scenario_list
        self.scale_factor = scale_factor

    def load_all(self):
        """
        Iterates through the city list, loads valid users, and concatenates tensors.
        Returns:
            master_X (Tensor): [N_Total, Tx, SC]
            master_y (Tensor): [N_Total] (LoS labels, unused for AE but kept for consistency)
        """
        all_chs = []
        all_labels = [] 
        
        print(f"Loading Multi-Scenario Dataset ({len(self.scenario_list)} cities)...")
        
        for scenario_name in self.scenario_list:
            print(f"  > Processing {scenario_name}...", end=" ")
            try:
                # 1. Get parameters & Generate
                params = get_parameters(scenario_name) 
                deepmimo_data = DeepMIMOv3.generate_data(params)
                
                # 2. Filter Valid Users
                los = deepmimo_data[0]['user']['LoS']
                valid_idx = np.where(los != -1)[0]
                
                raw_chs = deepmimo_data[0]['user']['channel'][valid_idx]
                
                # 3. Squeeze Rx Dim (1 -> None)
                if raw_chs.ndim == 4:
                    raw_chs = raw_chs.squeeze(axis=1)
                
                # 4. Convert & Scale
                t_chs = torch.tensor(raw_chs, dtype=torch.complex64) * self.scale_factor
                t_lbl = torch.tensor(los[valid_idx], dtype=torch.long)
                
                all_chs.append(t_chs)
                all_labels.append(t_lbl)
                print(f"Success. (+{len(t_chs)} samples)")
                
            except Exception as e:
                print(f"FAILED! Error: {e}")
                
        # 5. Concatenate & Shuffle
        if not all_chs:
            raise ValueError("No data loaded from any scenario!")
            
        master_X = torch.cat(all_chs, dim=0)
        master_y = torch.cat(all_labels, dim=0)
        
        # Shuffle
        perm = torch.randperm(len(master_X))
        
        print(f"--- Total Dataset: {len(master_X)} samples. Shape: {master_X.shape} ---")
        return master_X[perm], master_y[perm]

class NMSELoss(nn.Module):
    """
    Normalized Mean Squared Error (NMSE) Loss.
    NMSE = ||H - H_hat||^2 / ||H||^2
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_recon, x_target):
        # Error Power: || H - H_hat ||^2
        # Sum over (Channels, H, W) -> dims [1, 2, 3] if input is 4D
        # For complex input [B, Tx, SC], we handle real/imag separate or together.
        # Here input is expected to be stacked real/imag: [B, 2, Tx, SC]
        
        diff = x_target - x_recon
        error_power = torch.sum(diff**2, dim=[1, 2, 3]) 
        
        # Signal Power: || H ||^2
        sig_power = torch.sum(x_target**2, dim=[1, 2, 3])
        
        # NMSE per sample
        nmse = error_power / (sig_power + 1e-12)
        
        return torch.mean(nmse)

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, config, ref_power):
    """
    Training loop with Physics-Compliant Denoising.
    
    Args:
        ref_power (float): System reference power derived from training set.
                           Used to calculate fixed noise floor.
    """
    device = config.device
    model.to(device)
    
    # 1. Setup Logging
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_id = f"{config.task_name}_{config.latent_dim}"
    results_folder = os.path.join(config.results_dir, task_id, time_now)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(config.models_dir, exist_ok=True)
    
    log_file = os.path.join(results_folder, "training_log.csv")
    model_save_path = os.path.join(config.models_dir, f"csi_autoencoder_{config.latent_dim}.pth")
    
    print(f"\nStarting Training: {task_id}")
    print(f"Ref Power: {ref_power:.6e}")
    
    # Initialize CSV
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train NMSE", "Val NMSE", "Learning Rate", "Time"])

    # 2. Optimizer
    opt = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=20, cooldown=5, min_lr=1e-6
    )
    criterion = NMSELoss()
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # --- TRAIN STEP ---
        model.train()
        train_loss_sum = 0
        
        for batch in train_loader:
            x_clean = batch[0].to(device)
            
            # Physics-Compliant Denoising Training
            # We want the AE to learn to denoise a range of SNRs.
            # Sample SNR uniformly from [0, 20] dB for this batch.
            batch_snr_db = np.random.uniform(0, 20)
            
            # Calculate Noise Floor based on Fixed Reference Power
            snr_linear = 10 ** (batch_snr_db / 10.0)
            noise_power = ref_power / snr_linear
            
            # Apply Noise
            x_noisy = apply_awgn(x_clean, noise_power)
            
            # Forward & Loss
            _, x_recon_complex = model(x_noisy)
            
            # Convert to stacked real/imag for NMSE loss calculation
            recon_stacked = torch.stack([x_recon_complex.real, x_recon_complex.imag], dim=1)
            clean_stacked = torch.stack([x_clean.real, x_clean.imag], dim=1)
            
            loss = criterion(recon_stacked, clean_stacked)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # --- VALIDATION STEP ---
        model.eval()
        val_loss_sum = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_clean = batch[0].to(device)
                
                # Test at a fixed SNR (e.g., 10dB) for consistent validation metric
                val_snr_linear = 10 ** (10.0 / 10.0)
                val_noise_power = ref_power / val_snr_linear
                x_noisy_val = apply_awgn(x_clean, val_noise_power)
                
                _, x_recon_complex = model(x_noisy_val)
                
                recon_stacked = torch.stack([x_recon_complex.real, x_recon_complex.imag], dim=1)
                clean_stacked = torch.stack([x_clean.real, x_clean.imag], dim=1)
                
                loss = criterion(recon_stacked, clean_stacked)
                val_loss_sum += loss.item()
                
        avg_val_loss = val_loss_sum / len(val_loader)
        current_lr = opt.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        # --- LOGGING ---
        total_elapsed = time.time() - start_time
        time_str = f"{total_elapsed:.2f}s"
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Time: {time_str}")
        
        with open(log_file, mode='a', newline='') as file:
            csv.writer(file).writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", f"{current_lr:.1e}", time_str])
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            # print(f"\tNew Best Model Saved")

    # Finalization
    print("\nTraining Complete.")
    print(f"Loading best weights from {model_save_path}...")
    model.load_state_dict(torch.load(model_save_path))
    
    return model, history, results_folder

def plot_training_history(history, save_dir=None):
    """Plots Training and Validation NMSE."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], label='Training NMSE')
    plt.plot(epochs, history['val_loss'], label='Validation NMSE')
    
    plt.title('CSI Autoencoder Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('NMSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "training_curve.png"))
        print("Training plot saved.")
    else:
        plt.show()

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    config = AutoEncoderConfig()
    
    # 1. Load Data
    train_gen = MultiCityGenerator(config.train_cities, config.scale_factor)
    val_gen = MultiCityGenerator(config.val_cities, config.scale_factor)
    
    X_train, _ = train_gen.load_all()
    X_val, _ = val_gen.load_all()
    
    train_dl = DataLoader(TensorDataset(X_train), batch_size=config.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val), batch_size=config.batch_size, shuffle=False)
    
    # 2. Physics Calibration (Global Reference Power)
    # Calculate average power of the training set to set the Noise Floor baseline
    ref_power = torch.mean(torch.abs(X_train)**2).item()
    
    # 3. Model
    model = CSIAutoEncoder(latent_dim=config.latent_dim, mode="train")
    
    # 4. Train
    best_model, history, res_folder = train_model(model, train_dl, val_dl, config, ref_power)
    
    # 5. Visualize
    plot_training_history(history, save_dir=res_folder)