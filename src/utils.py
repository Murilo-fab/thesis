import ast

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset

import warnings
warnings.filterwarnings("ignore", message="Length of split at index")

def prepare_loaders(channels_tensor: torch.Tensor,
                    tokens_tensor: torch.Tensor,
                    split: list[int] = [0.7, 0.2, 0.1],
                    batch_size: int = 32,
                    seed: int = None):
    """
    Creates a train, validation and test loader;

    Inputs:
    channels_tensor (torch.Tensor): The raw channels tensor
    tokens_tensor (torch.Tensor): The raw tokens tensor
    split (list[int]): The split ratio between train, validation and test
    batch_size (int): Size of each batch
    seed (int): Seed for the generator

    Outputs:
    train_loader, val_loader, test_loader (DataLoader)
    """
    base_dataset = TensorDataset(channels_tensor, tokens_tensor)
    
    if seed is not None:
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, test_subset = random_split(base_dataset, split, generator)
    else:
        train_subset, val_subset, test_subset = random_split(base_dataset, split)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_subset(data_loader: DataLoader,
               ratio: float,
               batch_size: int=None,
               seed: int=None):
    """
    Returns a fraction of the original DataLoader

    Inputs:
    data_loader (DataLoader)
    ratio (float)
    batch_size (int): Size of each batch
    seed (int): Seed for the generator

    Outputs:
    fraction_data_loader (DataLoader)
    """
    dataset = data_loader.dataset
    n_samples = len(dataset)
    n_subset = max(1, int(ratio * n_samples))

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    indices = torch.randperm(n_samples, generator=generator)[:n_subset]
    subset = Subset(dataset, indices)

    batch_size = batch_size or data_loader.batch_size

    return DataLoader(subset,
                      batch_size=batch_size,
                      shuffle=True)


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