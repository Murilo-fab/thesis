import os
import subprocess

import torch
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader

import DeepMIMOv3
import numpy as np
from thesis.scenario_props import *

import warnings
warnings.filterwarnings("ignore", message="Length of split at index")

def get_parameters(scenario):
    """ Helper to get robust DeepMIMO parameters """
    # Default Configs
    N_ANT = 32
    N_SUB = 32
    SCS = 30e3
    DEFAULT_NUM_PATHS = 20

    # 1. Retrieves scenario-specific properties (e.g., antenna counts)
    scenario_configs = scenario_prop()

    # 2. Start with default DeepMIMO parameters
    params = DeepMIMOv3.default_params()

    # 3. Basic configuration
    params['dataset_folder'] = '../scenarios'
    params['scenario'] = scenario.split("_v")[0]

    # BS Selection Logic
    if scenario in ['city_18_denver', 'city_15_indianapolis']:
        params['active_BS'] = np.array([3])
    else:
        params['active_BS'] = np.array([1])

    params['enable_BS2BS'] = False
    params['num_paths'] = DEFAULT_NUM_PATHS

    n_ant_bs = N_ANT
    n_subcarriers = N_SUB
    scs = SCS
    
    params['bs_antenna']['shape'] = np.array([n_ant_bs, 1]) 
    params['bs_antenna']['rotation'] = np.array([0,0,-135])
    params['ue_antenna']['shape'] = np.array([1, 1])
    
    # 4. Scenario-specific configuration
    max_rows = scenario_configs.get(scenario, {'n_rows': 50})['n_rows']
    params['user_rows'] = np.arange(max_rows)
    
    # 5. OFDM configuration
    
    params['OFDM']['subcarriers'] = n_subcarriers
    params['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    params['OFDM']['bandwidth'] = scs * n_subcarriers / 1e9
    
    return params

def create_dataloaders(inputs, labels=None, train_ratio=0.8, val_ratio=0.2, batch_size=32, seed=42):
    """
    Splits data into Train/Val sets and creates DataLoaders.
    Supports partial usage (e.g., using only 1% of data for training).
    
    Args:
        inputs (Tensor): Input features (e.g., Channel Matrices).
        labels (Tensor, optional): Targets (e.g., LoS/NLoS). Can be None for unsupervised.
        train_ratio (float): Fraction of TOTAL data to use for Training (0.0 to 1.0).
        val_ratio (float): Fraction of TOTAL data to use for Validation.
        batch_size (int): Batch size for loaders.
        seed (int): Random seed for reproducible splitting.
        
    Returns:
        train_loader, val_loader
    """
    total_samples = len(inputs)
    
    # 1. Validation Check
    if train_ratio + val_ratio > 1.0:
        raise ValueError(f"Ratios sum to {train_ratio + val_ratio:.2f}, which exceeds 1.0")
        
    # 2. Determine Split Sizes
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    
    # 3. Shuffle Indices (Reproducible)
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(total_samples, generator=g)
    
    # 4. Slice Indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    # The rest (indices[n_train + n_val:]) are effectively discarded
    
    # 5. Create Subsets
    x_train = inputs[train_idx]
    x_val = inputs[val_idx]
    
    if labels is not None:
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        
        # Create Datasets with Labels
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
    else:
        # Create Datasets without Labels (for Unsupervised AE)
        train_ds = TensorDataset(x_train)
        val_ds = TensorDataset(x_val)
        
    # 6. Create Loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl

def clone_scenarios(scenario_name: str,
                    repo_url: str,
                    base_dir: str = ".",) -> None:
    """
    Clones specific DeepMIMO scenarios from a GitHub or Hugging Face repository using sparse checkout.
    
    This avoids downloading the entire dataset history, fetching only the requested scenario folder.

    Inputs:
        scenario_name (str): The name of the specific scenario folder to clone (e.g., "O1_60").
        repo_url (str): The URL of the Git repository.
        base_dir (str): The local directory where the 'scenarios' folder will be created.

    Outputs:
        None
    """
    # 1. Setup paths
    scenarios_path = os.path.join(base_dir, "scenarios")
    if not os.path.exists(scenarios_path):
        os.makedirs(scenarios_path)

    scenario_path = os.path.join(scenarios_path, scenario_name)

    # 2. Initialize sparse checkout if not already a git repo
    if not os.path.exists(os.path.join(scenarios_path, ".git")):
        print(f"Initializing sparse checkout in {scenarios_path}...")
        # Clone only the root level initially
        subprocess.run(["git", "clone", "--sparse", repo_url, "."], cwd=scenarios_path, check=True)
        subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=scenarios_path, check=True)
        # Install Git LFS hooks
        subprocess.run(["git", "lfs", "install"], cwd=scenarios_path, check=True)

    # 3. Add the requested scenario folder to the sparse checkout list
    print(f"Adding {scenario_name} to sparse checkout...")
    subprocess.run(["git", "sparse-checkout", "add", scenario_name], cwd=scenarios_path, check=True)

    # 4. Pull actual content (including large files)
    subprocess.run(["git", "lfs", "pull"], cwd=scenarios_path, check=True)

    print(f"Successfully cloned {scenario_name} into {scenario_path}.")

def apply_awgn(x_complex, snr_db):
    """
    Applies Global Standard AWGN to a batch of complex tensors.

    Args:
        x_complex (Tensor): Input batch of shape (B, ...), e.g., (B, U, 32, 32)
        snr_db (float/int): The target SNR in decibels.
        
    Returns:
        Tensor: The noisy complex signal.
    """
    # 1. Calculate the global signal power per batch item
    dims = tuple(range(1, x_complex.ndim))
    sig_power = torch.mean(torch.abs(x_complex)**2, dim=dims, keepdim=True)
    
    # 2. Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10.0)

    # 3. Derive noise power and standard deviation
    # P_noise = P_signal / SNR
    noise_power = sig_power / (snr_linear + 1e-12)

    # In complex AWGN, the noise power is split equally between Real and Imag
    # std = sqrt(P_noise / 2)
    noise_std = torch.sqrt(noise_power / 2)

    # 4. Generate Complex Noise
    noise_real = torch.randn_like(x_complex.real) * noise_std
    noise_imag = torch.randn_like(x_complex.imag) * noise_std
    
    return x_complex + torch.complex(noise_real, noise_imag)

def extract_features(model, dataloader, device):
    """
    Runs inference but stops before the classification head to get latent features.
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            
            # Get features through the Wrapper
            if hasattr(model, 'get_features'):
                feats = model.get_features(bx).flatten(start_dim=1)
            else:
                # Fallback to standard forward pass if get_features isn't used
                feats = model(bx)
                
            all_features.append(feats.cpu().numpy())
            all_labels.append(by.cpu().numpy())
            
    return np.concatenate(all_features), np.concatenate(all_labels)

def get_flops_and_params(model, input_tensor, device):
    """
    Calculates theoretical computational cost.
    """
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    try:
        stats = summary(model, input_data=input_tensor, verbose=0)
        return {
                "MFLOPs": stats.total_mult_adds / 1e6,
                "Params_M": stats.total_params / 1e6
            }
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return {"MFLOPs": 0, "Params_M": 0}

def get_latency(model, input_tensor, device, n_repeat=500):
    """
    Measures Encoder vs Head latency separately.
    """
    input_tensor = input_tensor.to(device)
    # 1. Warmup
    with torch.no_grad():
        for _ in range(50):
            feats = model.get_features(input_tensor)
            _ = model.task_head(feats)

    # 2. Setup Timers
    start = torch.cuda.Event(enable_timing=True)
    mid   = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    enc_times = []
    head_times = []

    # 3. Measurement Loop
    with torch.no_grad():
        for _ in range(n_repeat):
            start.record()

            # Encoder pass
            feat = model.get_features(input_tensor)

            mid.record()

            # Head pass
            _ = model.task_head(feat)

            end.record()

            torch.cuda.synchronize()

            enc_times.append(start.elapsed_time(mid))
            head_times.append(mid.elapsed_time(end))

    return {
        "Encoder_ms": np.mean(enc_times),
        "Head_ms": np.mean(head_times)
    }