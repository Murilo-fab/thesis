import os
import json

import torch

from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Union, List

@dataclass
class TaskConfig:
    # Identity & Scenario
    task_name: str
    scenario_name: str = "city_6_miami"

    # Ratios & SNR (Using default_factory for mutable lists)
    train_ratios: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8])
    snr_range: List[int] = field(default_factory=lambda: [-20, -15, -10, -5,  0, 5, 10, 15, 20])
    task_complexity: Optional[List[int]] = None

    # Hyperparameters
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    results_dir: str = "./results"

    def save_json(self, file_path: Optional[str] = None):
        """Saves config to the specified folder."""
        # Ensure the directory exists
        directory = os.path.dirname(file_path)

        # Create directory if it doesn't exist (and doesn't error if it's the current dir)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load_json(cls, file_path: str):
        """Loads a config from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No config found at {file_path}")
            
        with open(file_path, "r") as f:
            data = json.load(f)
            
        return cls(**data)

    def __post_init__(self):
        """Standardize paths and hardware after initialization."""
        self.results_dir = self.results_dir.rstrip("/")
        
        # Hardware fallback
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            
        # Example validation for task_complexity
        if self.task_complexity is not None:
            if not all(isinstance(x, int) for x in self.task_complexity):
                raise ValueError("task_complexity must be a list of integers.")


@dataclass
class ModelConfig:
    # 1. Metadata
    name: str

    # 2. Architecture Selection
    # encoder_type: "AE", "LWM", or None (for "raw")
    encoder_type: Optional[Literal["AE", "LWM"]] = None

    # mode: determines the forward pass logic in the Wrapper
    mode: Literal["raw", "ae", "cls", "channel_emb"] = "raw"

    # 3. Dimensions & Paths
    input_size: int = 2048
    output_size: int = 2
    latent_dim: Optional[int] = None
    weights_path: Optional[str] = None

    # 4. Task Configuration
    task_type: Literal["classification", "enhanced_classification, ""regression"] = "classification"

    # 5. Fine-tuning Logic
    # Accept None (frozen), "full" (all), or list of substrings (partial)
    fine_tune_layers: Union[List[str], Literal["full"], None] = None

    def save_json(self, file_path: str):
        """Serializes the configuration to a JSON file."""
        # Ensure the directory exists
        directory = os.path.dirname(file_path)

        # Create directory if it doesn't exist (and doesn't error if it's the current dir)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load_json(cls, file_path: str):
        """Creates a ModelConfig instance from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        return cls(**data)
    
    def __post_init__(self):
        """Simple validation to catch mismatched configs early."""
        if self.mode == "ae" and self.encoder_type != "AE":
            raise ValueError("Mode 'ae' requires encoder_type='AE'")
        
        if self.mode in ["cls", "channel_emb"] and self.encoder_type != "LWM":
            raise ValueError(f"Mode '{self.mode}' requires encoder_type='LWM'")
            
        if self.encoder_type == "AE" and self.latent_dim is None:
            raise ValueError("AutoEncoder requires a latent_dim.")
