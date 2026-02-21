"""
Utility Functions for Wireless Channel Deep Learning.

This module provides helper functions for:
1. DeepMIMO Configuration & Data Loading.
2. Physics-based Simulation (AWGN, Channel Noise).
3. Data Splitting & Loader Creation.
4. Model Analysis (FLOPs, Latency, Feature Extraction).

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

# --- 1. Standard Library Imports ---
import os
import subprocess
import warnings
from typing import Tuple, Dict, Any, Optional

# --- 2. Third-Party Imports ---
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchinfo import summary

# --- 3. Local Imports ---
import DeepMIMOv3
# Assuming 'thesis.scenario_props' contains a dictionary of scenario metadata
from thesis.scenario_props import scenario_prop 

# Suppress specific PyTorch warnings that clutter logs
warnings.filterwarnings("ignore", message="Length of split at index")

# =============================================================================
# PART 1: DEEPMIMO & DATA HANDLING
# =============================================================================

def get_parameters(scenario: str) -> Dict[str, Any]:
    """
    Constructs a robust parameter dictionary for the DeepMIMOv3 engine.
    
    This function:
    1. Loads scenario-specific properties (e.g., max rows, antenna counts).
    2. Sets default physical parameters (32 antennas, 32 subcarriers).
    3. Selects the correct Base Station (BS) index based on the city.
    4. Configures OFDM parameters.

    Args:
        scenario (str): The name of the scenario (e.g., 'city_18_denver').

    Returns:
        dict: A configuration dictionary ready for `DeepMIMOv3.generate_data()`.
    """
    # Constants
    N_ANT = 32
    N_SUB = 32
    SCS = 30e3  # Subcarrier Spacing (Hz)
    DEFAULT_NUM_PATHS = 20

    # 1. Retrieve metadata
    scenario_configs = scenario_prop()
    
    # 2. Initialize defaults
    params = DeepMIMOv3.default_params()
    params['dataset_folder'] = '../scenarios'
    params['scenario'] = scenario.split("_v")[0] # Handle version suffixes

    # 3. BS Selection Logic (Scenario Dependent)
    if scenario in ['city_18_denver', 'city_15_indianapolis']:
        params['active_BS'] = np.array([3])
    else:
        params['active_BS'] = np.array([1])

    # 4. Antenna & Channel Config
    params['enable_BS2BS'] = False
    params['num_paths'] = DEFAULT_NUM_PATHS
    
    params['bs_antenna']['shape'] = np.array([N_ANT, 1]) 
    params['bs_antenna']['rotation'] = np.array([0, 0, -135]) # Standard sector orientation
    params['ue_antenna']['shape'] = np.array([1, 1])          # Single Antenna UE
    
    # 5. User Grid Config
    # Default to 50 rows if not specified in props
    max_rows = scenario_configs.get(scenario, {'n_rows': 50})['n_rows']
    params['user_rows'] = np.arange(max_rows)
    
    # 6. OFDM Config
    params['OFDM']['subcarriers'] = N_SUB
    params['OFDM']['selected_subcarriers'] = np.arange(N_SUB)
    params['OFDM']['bandwidth'] = SCS * N_SUB / 1e9 # GHz
    
    return params

def clone_scenarios(scenario_name: str, repo_url: str, base_dir: str = ".") -> None:
    """
    Clones specific DeepMIMO scenarios using Git Sparse Checkout.
    
    This is bandwidth-efficient: it downloads ONLY the requested scenario folder
    instead of the entire history of all scenarios.

    Args:
        scenario_name (str): Folder name to clone (e.g., 'O1_60').
        repo_url (str): Git repository URL.
        base_dir (str): Local parent directory for the 'scenarios' folder.
    """
    scenarios_path = os.path.join(base_dir, "scenarios")
    if not os.path.exists(scenarios_path):
        os.makedirs(scenarios_path)

    # Initialize Sparse Checkout if new
    if not os.path.exists(os.path.join(scenarios_path, ".git")):
        print(f"Initializing sparse checkout in {scenarios_path}...")
        subprocess.run(["git", "clone", "--sparse", repo_url, "."], cwd=scenarios_path, check=True)
        subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=scenarios_path, check=True)
        subprocess.run(["git", "lfs", "install"], cwd=scenarios_path, check=True)

    # Add requested folder
    print(f"Adding {scenario_name} to sparse checkout...")
    subprocess.run(["git", "sparse-checkout", "add", scenario_name], cwd=scenarios_path, check=True)
    
    # Pull LFS files (large datasets)
    subprocess.run(["git", "lfs", "pull"], cwd=scenarios_path, check=True)
    print(f"Successfully cloned {scenario_name}.")

# =============================================================================
# PART 2: DATA SPLITTING & LOADERS
# =============================================================================

def create_dataloaders(
    inputs: torch.Tensor, 
    labels: Optional[torch.Tensor] = None, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.2, 
    test_ratio: float = 0.1, 
    batch_size: int = 32, 
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits tensors into Train/Val/Test sets and wraps them in DataLoaders.
    
    Args:
        inputs (Tensor): Input features [N, ...].
        labels (Tensor, optional): Targets [N, ...]. If None, creates TensorDataset(inputs).
        train_ratio (float): Fraction for training (e.g., 0.7).
        val_ratio (float): Fraction for validation.
        test_ratio (float): Fraction for testing.
        batch_size (int): Mini-batch size.
        seed (int): Seed for reproducible random splitting.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    total_samples = len(inputs)
    
    # Sanity Check
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(f"Ratios sum to {train_ratio + val_ratio + test_ratio:.2f}, must be 1.0")
        
    # Calculate Split Sizes
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = int(total_samples * test_ratio)
    
    # Reproducible Shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(total_samples, generator=g)
    
    # Slicing
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]
    
    # Create Subsets
    x_train, x_val, x_test = inputs[train_idx], inputs[val_idx], inputs[test_idx]
    
    if labels is not None:
        y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        test_ds = TensorDataset(x_test, y_test)
    else:
        # Unsupervised Case (Autoencoders)
        train_ds = TensorDataset(x_train)
        val_ds = TensorDataset(x_val)
        test_ds = TensorDataset(x_test)
        
    # Create Loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl, test_dl

def get_subset(original_loader: DataLoader, ratio: float, seed: int = 42) -> DataLoader:
    """
    Creates a new DataLoader containing a random subset (x%) of the original data.
    Useful for Data Efficiency experiments (Training on 1%, 10%, etc.).
    
    Args:
        original_loader (DataLoader): The source loader.
        ratio (float): Percentage to keep (0.0 < ratio <= 1.0).
        seed (int): Seed for reproducibility.
        
    Returns:
        DataLoader: A new loader iterating over the subset.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio}")

    dataset = original_loader.dataset
    total_samples = len(dataset)
    subset_size = int(total_samples * ratio)
    
    # Random Selection
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(total_samples, generator=g).tolist()
    subset_indices = indices[:subset_size]
    
    # Create Subset
    subset_ds = Subset(dataset, subset_indices)

    # Drop last verification
    batch_size = original_loader.batch_size
    drop_last = (len(subset_ds) % batch_size) == 1
    
    # Preserve original loader settings (workers, pinning)
    new_loader = DataLoader(
        subset_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        drop_last=drop_last
    )
    
    return new_loader

# =============================================================================
# PART 3: PHYSICS & SIMULATION
# =============================================================================

def apply_awgn(x_complex: torch.Tensor, noise_power: float) -> torch.Tensor:
    """
    Applies Complex Gaussian Noise (AWGN) with a fixed variance.
    
    Using fixed noise_power (derived from reference signal power) ensures correct 
    simulation of Path Loss. Far-away users (low signal) naturally get lower SNR 
    than close users (high signal) when noise floor is constant.
    
    Args:
        x_complex (Tensor): Input signal [B, ...].
        noise_power (float): Variance of the noise (N0).
        
    Returns:
        Tensor: Noisy signal.
    """
    # 1. Calculate Standard Deviation
    # Power splits equally into Real and Imaginary parts (P_total = P_real + P_imag)
    # std = sqrt(Power / 2)
    noise_std = torch.sqrt(torch.tensor(noise_power, device=x_complex.device) / 2.0)

    # 2. Generate Noise
    noise_real = torch.randn_like(x_complex.real) * noise_std
    noise_imag = torch.randn_like(x_complex.imag) * noise_std
    
    # 3. Add to Signal
    return x_complex + torch.complex(noise_real, noise_imag)

# =============================================================================
# PART 4: MODEL ANALYSIS & METRICS
# =============================================================================

def extract_features(model: torch.nn.Module, dataloader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference to extract latent features (before the final classification head).
    Used for t-SNE visualization.
    
    Args:
        model: Trained PyTorch model.
        dataloader: Data to extract features from.
        device: 'cuda' or 'cpu'.
        
    Returns:
        (features, labels): Numpy arrays of extracted embeddings and corresponding labels.
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            
            # Wrapper models usually expose a 'get_features' method
            if hasattr(model, 'get_features'):
                feats = model.get_features(bx)
                # Flatten spatial dims if CNN (Batch, C, H, W) -> (Batch, Features)
                feats = feats.flatten(start_dim=1)
            else:
                # Fallback: Just run forward pass (if model outputs features directly)
                feats = model(bx)
                
            all_features.append(feats.cpu().numpy())
            all_labels.append(by.cpu().numpy())
            
    return np.concatenate(all_features), np.concatenate(all_labels)

def get_flops_and_params(model: torch.nn.Module, input_tensor: torch.Tensor, device: str) -> Dict[str, float]:
    """
    Calculates theoretical computational cost (FLOPs) and Parameter count.
    
    Args:
        model: PyTorch model.
        input_tensor: A sample input tensor (to determine input shape).
        
    Returns:
        dict: {"MFLOPs": float, "Params_M": float}
    """
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    
    try:
        # torchinfo provides a clean summary
        stats = summary(model, input_data=input_tensor, verbose=0)
        return {
            "MFLOPs": stats.total_mult_adds / 1e6,
            "Params_M": stats.total_params / 1e6
        }
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return {"MFLOPs": 0.0, "Params_M": 0.0}

def get_latency(model: torch.nn.Module, input_tensor: torch.Tensor, device: str, n_repeat: int = 500) -> Dict[str, float]:
    """
    Measures CUDA Latency for the Encoder and the Task Head separately.
    
    Args:
        model: PyTorch wrapper model (must have .get_features() and .task_head()).
        input_tensor: Sample input.
        device: Must be 'cuda'.
        n_repeat: Number of iterations for averaging.
        
    Returns:
        dict: {"Encoder_ms": float, "Head_ms": float}
    """
    if device != "cuda":
        # Latency on CPU is unreliable/variable due to OS scheduling
        return {"Encoder_ms": 0.0, "Head_ms": 0.0}

    input_tensor = input_tensor.to(device)
    
    # 1. Warmup (Wake up GPU)
    with torch.no_grad():
        for _ in range(50):
            if hasattr(model, 'get_features'):
                feats = model.get_features(input_tensor)
                if hasattr(model, 'task_head'):
                    _ = model.task_head(feats)

    # 2. Setup CUDA Events
    start = torch.cuda.Event(enable_timing=True)
    mid   = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    enc_times = []
    head_times = []

    # 3. Measurement Loop
    with torch.no_grad():
        for _ in range(n_repeat):
            start.record()

            # Measure Encoder
            feat = model.get_features(input_tensor) if hasattr(model, 'get_features') else model(input_tensor)

            mid.record()

            # Measure Head (if exists)
            if hasattr(model, 'task_head'):
                _ = model.task_head(feat)

            end.record()

            torch.cuda.synchronize()

            enc_times.append(start.elapsed_time(mid))
            head_times.append(mid.elapsed_time(end))

    return {
        "Encoder_ms": np.mean(enc_times),
        "Head_ms": np.mean(head_times)
    }