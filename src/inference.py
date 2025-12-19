from typing import Optional, Literal

# Torch imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, Tensor

import warnings
warnings.filterwarnings('ignore')

def lwm_inference(
    model: Optional[nn.Module],
    data: Tensor,
    input_type: Literal["raw", "cls_emb", "channel_emb"] = "cls_emb",
    device: str = "cpu",
    batch_size: int = 64
) -> Tensor:
    """
    Performs inference using the LWM model to generate embeddings.

    Inputs:
    model (torch.nn.Module): The pre-trained model for inference. Can be None if input_type is 'raw'.
    data (torch.Tensor): The input data tensor.
    input_type (str): The type of output to generate.
    device (str): The device to run inference on ('cpu' or 'cuda').
    batch_size (int): The batch size for the dataloader.

    Outputs:
    output (torch.Tensor): The resulting tensor from the inference process.
    """
    if input_type == "raw":
        return torch.tensor(data).float()

    if model is None:
        raise ValueError("A model must be provided for inference when input_type is not 'raw'.")

    model.to(device)
    model.eval()

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            output, _ = model(input_ids)

            if input_type == "cls_emb":
                batch_embeddings = output[:, 0, :]
            elif input_type == "channel_emb":
                batch_embeddings = output[:, 1:, :]
            else:
                raise ValueError(f"Invalid input_type: '{input_type}'. Must be 'cls_emb' or 'channel_emb'.")
            
            embeddings.append(batch_embeddings)
            
    return torch.cat(embeddings, dim=0).float()
