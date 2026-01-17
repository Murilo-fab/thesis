import os
import subprocess
from typing import TypedDict, List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset

import warnings
warnings.filterwarnings("ignore", message="Length of split at index")

class OptimizerConfigs(TypedDict):
    """
    Type definition for Optimizer configuration parameters.

    Attributes:
        task_head_lr (float): Learning rate for the task-specific head.
        encoder_lr (Optional[float]): Learning rate for the encoder (if applicable).
    """
    task_head_lr: float
    encoder_lr: Optional[float]

def prepare_loaders(channels_tensor: torch.Tensor,
                    split: List[float] = [0.7, 0.2, 0.1],
                    batch_size: int = 32,
                    seed: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates a train, validation and test loader;

    Inputs:
        channels_tensor (torch.Tensor): The raw input features (channels).
        tokens_tensor (torch.Tensor): The raw target labels (tokens).
        split (List[float]): The split ratios for [train, val, test]. Must sum to 1.0.
        batch_size (int): The number of samples per batch.
        seed (int, optional): Random seed for reproducibility.

    Outputs:
        Tuple[DataLoader, DataLoader, DataLoader]: The train, validation, and test loaders.
    """
    # 1. Create a TensorDataset wrapping the inputs and targets
    base_dataset = TensorDataset(channels_tensor)#, tokens_tensor)
    
    # 2. Define the generator for reproducibility
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator() # Default generator

    # 3. Calculate lengths based on ratios
    # random_split requires integer lengths, not float ratios
    total_len = len(base_dataset)
    lengths = [int(r * total_len) for r in split]
    
    # Fix rounding errors: Add the remainder to the first split (train) to ensure sum matches total
    lengths[0] += total_len - sum(lengths)

    # 4. Perform the split
    if seed is not None:
         train_subset, val_subset, test_subset = random_split(base_dataset, lengths, generator=generator)
    else:
         train_subset, val_subset, test_subset = random_split(base_dataset, lengths)

    # 5. Create DataLoaders
    # Train is shuffled, validation/test are not
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_subset(data_loader: DataLoader,
               ratio: float,
               batch_size: Optional[int] = None,
               seed: Optional[int] = None) -> DataLoader:
    """
    Returns a new DataLoader containing a random fraction of the original data.

    Inputs:
        data_loader (DataLoader): The original source DataLoader.
        ratio (float): The fraction of data to keep (0.0 to 1.0).
        batch_size (int, optional): The batch size for the new loader. Defaults to original.
        seed (int, optional): Random seed for selecting indices.

    Outputs:
        DataLoader: A new DataLoader containing the sampled subset.
    """
    # 1. Access the underlying dataset
    dataset = data_loader.dataset
    n_samples = len(dataset)
    n_subset = max(1, int(ratio * n_samples))

    # 2. Determine indices for the subset
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Random permutation of indices, taking the first n_subset
    indices = torch.randperm(n_samples, generator=generator)[:n_subset]
    subset = Subset(dataset, indices)

    # 3. Determine batch size (use new one or fallback to original)
    final_batch_size = batch_size if batch_size is not None else data_loader.batch_size

    # 4. Return new loader
    return DataLoader(subset,
                      batch_size=final_batch_size,
                      shuffle=True)


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

def calculate_noise_power(bandwidth_ghz, noise_figure_db=9):
    """
    Calculates noise variance (sigma^2) in linear scale (Watts).
    """
    k_B = 1.380649e-23  # Boltzmann constant
    T = 290             # Temperature (Kelvin)
    BW_Hz = bandwidth_ghz * 1e9 # Convert GHz to Hz
    
    # Thermal Noise Density (N0)
    noise_spectral_density = k_B * T 
    
    # Noise Figure in Linear Scale
    noise_figure_linear = 10 ** (noise_figure_db / 10)
    
    # Total Noise Power
    noise_power_watts = noise_spectral_density * BW_Hz * noise_figure_linear
    
    return noise_power_watts
