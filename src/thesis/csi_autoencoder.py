import csv
import os
import time
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import DeepMIMOv3
from thesis.utils import get_parameters

class RefineBlock(nn.Module):
    """
    A Residual Block that refines features without changing their size.
    Input: (B, C, H, W) -> Output: (B, C, H, W)
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
        # Add residual connection
        return self.relu(out + residual)

class CSIAutoEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # Encoder
        self.input_norm = nn.BatchNorm2d(2)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # Extra depth
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32->16
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16->8
            
            nn.Flatten()
        )
        
        # Calculate flat size (128 * 8 * 8)
        self.flat_dim = 128 * 8 * 8
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        
        self.decoder_initial = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 8->16
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.refine1 = nn.Sequential(
            RefineBlock(64),
            RefineBlock(64)
        )
        
        self.decoder_final = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=1), # 16->32
        )

    def load_weights(self, path, device="cpu"):
        """
        Safely loads model weights from a .pth file.
        
        Args:
            path (str): Path to the .pth file.
            device (str): 'cpu' or 'cuda'.
        """
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")
            
        try:
            # map_location allows loading GPU-trained models on CPU
            state_dict = torch.load(path, map_location=device)
            
            # Handle standard vs DataParallel keys (remove 'module.' prefix if present)
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
            self.load_state_dict(state_dict)
            self.to(device)
            self.eval()
            
        except RuntimeError as e:
            print(f"Error loading weights: {e}")
            print(f"Ensure the model latent_dim matches the checkpoint.")
            raise e

    def forward(self, x):
        # 1. Norm
        x_real = torch.stack([x.real, x.imag], dim=1).float()
        x_norm = self.input_norm(x_real)
        
        # 2. Encode
        feat = self.encoder(x_norm)
        z = self.fc_enc(feat)
        
        # 3. Decode
        # Expand Latent
        x_recon = self.fc_dec(z).view(-1, 128, 8, 8)
        
        # Upsample 1
        x_recon = self.decoder_initial(x_recon) # (B, 64, 16, 16)
        
        # Refine (Residual)
        x_recon = self.refine1(x_recon)         # (B, 64, 16, 16) - Cleaner
        
        # Upsample 2 (Final Projection)
        x_recon = self.decoder_final(x_recon)   # (B, 2, 32, 32)
        
        return z, torch.complex(x_recon[:,0], x_recon[:,1])
    
    
class CSIAEWrapper(nn.Module):
    def __init__(self, csi_ae_model, task_head):
        super().__init__()
        self.csi_ae_model = csi_ae_model
        self.task_head = task_head

        for param in self.csi_ae_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (Batch, 32, 32) OR (Batch, K_Users, 32, 32)
        Returns:
            Output from task_head (B, K, Dim) or (B, Dim)
        """
        # 1. Shape Handling (Flatten Multi-User)
        if x.ndim == 4:
            B, K, M, N = x.shape
            # Flatten to (B*K, M, N) for the standard Encoder
            encoder_input = x.view(B * K, M, N)
        else:
            encoder_input = x

        # 2. Extract Features
        features, _ = self.csi_ae_model(encoder_input)

        # 3. Restore Multi-User shape
        if x.ndim == 4:
            B, K, M, N = x.shape
            # Flatten to (B*K, M, N) for the standard Encoder
            # features = features.view(B, K, -1)
            features = features.view(B, -1)

        # 4. Task Head
        out = self.task_head(features)

        return out
    
@dataclass
class TrainingConfig:
    TASK_NAME = "csi_autoencoder_training"

    # 1. Curriculum
    # Train: Mix of 5 diverse cities to learn general physics
    TRAIN_CITIES = [
        "city_7_sandiego", 
        "city_11_santaclara", 
        "city_12_fortworth", 
        "city_15_indianapolis",
        "city_19_oklahoma" 
    ]
    
    # Val: Check convergence (Unseen during gradient updates)
    VAL_CITY = ["city_18_denver"] 
    
    # Test
    TEST_CITY = ["city_6_miami"]

    # 2. Physics & Model
    SCALE_FACTOR = 1e6
    LATENT_DIM = 64
    
    # 3. Training
    BATCH_SIZE = 128 
    EPOCHS = 400      
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODELS_DIR = "./models"
    RESULTS_BASE = "./results/"

class DeepMIMOGenerator:
    """
    Aggregates data from multiple DeepMIMO scenarios into a single dataset.
    """
    def __init__(self, scenario_list, scale_factor=1e6):
        self.scenario_list = scenario_list
        self.scale_factor = scale_factor

    def load_all(self):
        all_chs = []
        all_labels = [] # LoS/NLoS labels
        
        print(f"Loading Multi-Scenario Dataset ({len(self.scenario_list)} cities)")
        
        for scenario_name in self.scenario_list:
            print(f"\tProcessing {scenario_name}...", end=" ")
            try:
                # 1. Get parameters
                params = get_parameters(scenario_name) 
                
                # 2. Generate
                deepmimo_data = DeepMIMOv3.generate_data(params)
                
                # 3. Filter Valid
                los = deepmimo_data[0]['user']['LoS']
                valid_idx = np.where(los != -1)[0]
                
                raw_chs = deepmimo_data[0]['user']['channel'][valid_idx]
                
                # 4. Squeeze Rx Dim (if 1x32x32 -> 32x32)
                if raw_chs.ndim == 4:
                    raw_chs = raw_chs.squeeze(axis=1)
                
                # 5. Convert & Scale
                t_chs = torch.tensor(raw_chs, dtype=torch.complex64) * self.scale_factor
                t_lbl = torch.tensor(los[valid_idx], dtype=torch.long)
                
                all_chs.append(t_chs)
                all_labels.append(t_lbl)
                print(f"Done. (+{len(t_chs)} samples)")
                
            except Exception as e:
                print(f"FAILED! {e}")
                
        # 6. Concatenate everything
        if not all_chs:
            raise ValueError("No data loaded from any scenario!")
            
        master_X = torch.cat(all_chs, dim=0)
        master_y = torch.cat(all_labels, dim=0)
        
        # 7. Global Shuffle
        perm = torch.randperm(len(master_X))
        
        print(f"Total Dataset: {len(master_X)} samples. Shape: {master_X.shape} ---")
        return master_X[perm], master_y[perm]
    

def add_awgn(x_clean, min_snr_db=0.0, max_snr_db=20.0):
    """
    Adds Additive White Gaussian Noise (AWGN) to a batch of complex channels.
    
    Args:
        x_clean: (Batch, ...) Complex Tensor
        min_snr_db, max_snr_db: Range of SNR to sample from uniformly.
    """
    B = x_clean.shape[0]
    device = x_clean.device
    
    # 1. Random SNR for each sample in the batch
    snr_db = torch.empty(B, 1, 1, device=device).uniform_(min_snr_db, max_snr_db)
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 2. Calculate Signal Power per sample
    # Flatten dimensions for power calc: (Batch, ...) -> (Batch, -1)
    flat = x_clean.reshape(B, -1)
    sig_power = torch.mean(flat.abs()**2, dim=1).reshape(B, 1, 1)
    
    # 3. Calculate Noise Power & Std
    noise_power = sig_power / snr_linear
    # Divide by 2 because noise splits into Real and Imag parts
    noise_std = torch.sqrt(noise_power / 2)
    
    # 4. Generate Complex Noise
    noise = (torch.randn_like(x_clean.real) * noise_std) + \
            1j * (torch.randn_like(x_clean.imag) * noise_std)
            
    return x_clean + noise

class NMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_recon, x_target):
        """
        Calculates NMSE between reconstructed and target channels.
        
        Args:
            x_recon: (Batch, ...) Complex or Real Tensor
            x_target: (Batch, ...) Complex or Real Tensor
        """
        # 1. Calculate Error Power: || H - H_hat ||^2
        diff = x_target - x_recon
        # Sum over all dimensions except Batch (dim 0)
        # abs() works for both complex and real
        error_power = torch.sum(diff.abs()**2, dim=[1, 2, 3]) 
        
        # 2. Calculate Signal Power: || H ||^2
        sig_power = torch.sum(x_target.abs()**2, dim=[1, 2, 3])
        
        # 3. NMSE per sample
        # Add epsilon for stability (prevent division by zero for silence)
        nmse = error_power / (sig_power + 1e-10)
        
        # 4. Return Mean NMSE over batch
        return torch.mean(nmse)

def train_model(model, train_loader, val_loader, config):
    """
    Trains the Autoencoder with customized CSV logging and folder structures.
    """
    device = config.DEVICE
    model.to(device)
    
    # 1. Setup Directories & Paths
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Results Folder: e.g. results/csi_autoencoder_training_64/2023-10-25_14-30-00
    task_id = f"{config.TASK_NAME}_{config.LATENT_DIM}"
    results_folder = os.path.join(config.RESULTS_BASE, task_id, time_now)
    os.makedirs(results_folder, exist_ok=True)
    
    # Models Folder: Ensure the base model directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Paths
    log_file = os.path.join(results_folder, "training_log.csv")
    
    # Model Save Path: models/csi_autoencoder_64.pth
    model_save_path = os.path.join(config.MODELS_DIR, f"csi_autoencoder_{config.LATENT_DIM}.pth")
    
    print(f"Starting Training: {task_id}")
    print(f"Logging to: {log_file}")
    print(f"Best Model will be saved to: {model_save_path}")

    # 2. Initialize CSV
    headers = ["Epoch", "Train Loss", "Validation Loss", "Learning Rate", "Time"]
    
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    # 3. Optimization Setup
    opt = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, 
        mode='min', 
        factor=0.5, 
        patience=20,
        cooldown=5,
        min_lr=1e-6
    )
    criterion = NMSELoss()
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # TRAIN STEP
        model.train()
        train_loss_sum = 0
        
        for batch in train_loader:
            x_clean = batch[0].to(device)
            
            # Dynamic Denoising
            with torch.no_grad():
                x_noisy = add_awgn(x_clean, min_snr_db=0, max_snr_db=20)
            
            # Forward & Loss
            _, x_recon = model(x_noisy)
            loss = criterion(
                torch.stack([x_recon.real, x_recon.imag], dim=1), 
                torch.stack([x_clean.real, x_clean.imag], dim=1)
                )
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # VALIDATION STEP
        model.eval()
        val_loss_sum = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_clean = batch[0].to(device)
                
                # Fixed SNR for validation
                x_noisy_val = add_awgn(x_clean, min_snr_db=10, max_snr_db=10)
                
                _, x_recon = model(x_noisy_val)
                loss = criterion(
                    torch.stack([x_recon.real, x_recon.imag], dim=1), 
                    torch.stack([x_clean.real, x_clean.imag], dim=1)
                    )
                val_loss_sum += loss.item()
                
        avg_val_loss = val_loss_sum / len(val_loader)

        current_lr = opt.param_groups[0]['lr']

        scheduler.step(avg_val_loss)
        
        # LOGGING
        total_elapsed = time.time() - start_time
        time_str = f"{total_elapsed:.2f}s"

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1:02d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Time: {time_str}")
        
        # Write to CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,               # Epoch
                f"{avg_train_loss:.6f}", # Train Loss
                f"{avg_val_loss:.6f}",   # Validation Loss
                f"{current_lr:.1e}",     # Learning Rate
                time_str                 # Time
            ])
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"\tNew Best Model Saved (Val: {best_val_loss:.6f})")

    # FINALIZATION
    print("\nTraining Complete.")
    print(f"Loading best weights from {model_save_path}...")
    model.load_state_dict(torch.load(model_save_path))
    
    return model, history, results_folder

def plot_training_history(history, save_dir=None):
    """
    Plots the Training and Validation Loss (NMSE) from the history dictionary.
    
    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_loss' lists.
        save_dir (str, optional): Directory to save the plot image. If None, just shows it.
    """
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    
    # Plot Lines
    plt.plot(epochs, train_loss, 'b-o', label='Training NMSE', linewidth=2, markersize=4)
    plt.plot(epochs, val_loss, 'r-s', label='Validation NMSE', linewidth=2, markersize=4)
    
    # Styling
    plt.title('CSI Autoencoder Training Progress', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('NMSE Loss (Normalized)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.yscale('log') 
    
    plt.tight_layout()

    # Save or Show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()