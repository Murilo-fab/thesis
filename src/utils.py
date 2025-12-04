import ast

from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import warnings
warnings.filterwarnings("ignore", message="Length of split at index")

LN2 = torch.log(torch.tensor(2.0))
EPS = 1e-12

def load_lwm_model(base_model: torch.nn.Module, model_path: str, device: torch.device) -> torch.nn.Module:
    """Loads the pre-trained LWM model and prepares it for inference"""
    print("Loading LWM model...")
    model = base_model
    
    state_dict = torch.load(model_path, map_location=device)
    # Remove 'module.' prefix if the model was saved with DataParallel
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Use DataParallel if multiple GPUs are available 
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
        model = nn.DataParallel(model)

    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
    return model

def prepare_loaders(channels_tensor: torch.Tensor,
                    split: list[int] = [0.7, 0.2, 0.1],
                    batch_size: int = 32,
                    seed: int = None):
    
    base_dataset = TensorDataset(channels_tensor)
    
    if seed is not None:
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, test_subset = random_split(base_dataset, split, generator)
    else:
        train_subset, val_subset, test_subset = random_split(base_dataset, split)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_parameters(src: str):
    parameters = {}
    with open(src, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            key, value = line.split(":")

            key = key.strip()
            value = value.strip()

            try:
                value = ast.literal_eval(value)
            except Exception:
                pass

            parameters[key] = value

    return parameters