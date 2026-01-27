import os
import subprocess
from typing import TypedDict, Optional

import torch
from torch.utils.data import TensorDataset, DataLoader, Subset

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
    params['dataset_folder'] = './scenarios'
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

class OptimizerConfigs(TypedDict):
    """
    Type definition for Optimizer configuration parameters.

    Attributes:
        task_head_lr (float): Learning rate for the task-specific head.
        encoder_lr (Optional[float]): Learning rate for the encoder (if applicable).
    """
    task_head_lr: float
    encoder_lr: Optional[float]

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

def normalize_channels(x_complex):
    """
    Normalizes the (B, 32, 32) complex tensor.
    """
    # 1. View as floats to get global mean/std of the signal
    # (B, 32, 32) complex -> (B, 32, 32, 2) real -> flat
    all_values = torch.view_as_real(x_complex)
    mean = all_values.mean()
    std = all_values.std()

    # 2. Apply Normalization directly to Complex Tensor
    x_norm = (x_complex - mean) / (std + 1e-9)
    
    return x_norm

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    name: str
    encoder_type: str

    task_type: str
    latent_dim: int
    
    weights_path: Optional[str] = None
    # freeze_encoder: bool = True
    
    # Extra args for specific wrappers (e.g., tokenizer for LWM)
    extra_args: Dict[str, Any] = field(default_factory=dict)
    
    # Head configuration (e.g., num_classes=2)
    head_args: Dict[str, Any] = field(default_factory=lambda: {"num_classes": 2})

import torch

from thesis.csi_autoencoder import *
from thesis.lwm_model import * 
from thesis.downstream_models import *


def build_model_from_config(config: ModelConfig):
    """
    Factory function that instantiates a model based on the provided configuration.
    """
    # 1. Build task head
    input_dim = config.head_args['input_size']
    num_classes = config.head_args.get('num_classes', 2)
    
    if config.task_type == "classification":
        task_head = ClassificationHead(input_dim, num_classes)
    elif config.task_type == "regression":
        # Regression: Output 1 value, usually no softmax
        task_head = ClassificationHead(input_dim, 1) 
    else:
        return None

    # 2. Build backbone & wrap
    
    # Case A: Autoencoder
    if config.encoder_type == "AE":
        # Instantiate Base Architecture
        backbone = CSIAutoEncoder(latent_dim=config.latent_dim)
        
        # Load Weights (if provided, otherwise it's a Raw/Random AE)
        if config.weights_path:
            backbone.load_weights(config.weights_path)

        # Wrap
        model = CSIAEWrapper(
            csi_ae_model=backbone, 
            task_head=task_head
        )

    # Case B: LWM
    elif config.encoder_type == "LWM":
        # Configs
        tokenizer = Tokenizer(patch_rows=4, patch_cols=4, scale_factor=1e0)
        
        if config.weights_path:
            backbone = lwm.from_pretrained(config.weights_path)
        else:
            backbone = lwm()

        # Wrap
        model = LWMWrapper(
            tokenizer=tokenizer,
            lwm_model=backbone,
            task_head=task_head,
            mode=config.extra_args.get("lwm_mode", "cls"),
        
        )

        fine_tune_layers = config.extra_args.get("fine_tune_layers")

        if fine_tune_layers:
            model.unfreeze_layers(fine_tune_layers)

    # Case C: Raw Data
    elif config.encoder_type is None:
        # No backbone, just the head operating on flattened data
        # We use a simple wrapper to handle flattening if needed
        class LinearWrapper(torch.nn.Module):
            def __init__(self, head):
                super().__init__()
                self.head = head
            def forward(self, x):
                # Flatten (Batch, 32, 32) -> (Batch, 2048)
                if x.ndim > 2:
                    x = torch.hstack((x.real, x.imag)).flatten(start_dim=1)
                return self.head(x)
                
        model = LinearWrapper(task_head)

    else:
        raise ValueError(f"Unknown encoder_type: {config.encoder_type}")

    return model

