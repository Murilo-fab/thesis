import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
from typing import Optional, Literal
from torch import nn, Tensor

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

    Args:
        model: The pre-trained model for inference. Can be None if input_type is 'raw'.
        data: The input data tensor.
        input_type: The type of output to generate.
        device: The device to run inference on ('cpu' or 'cuda').
        batch_size: The batch size for the dataloader.

    Returns:
        The resulting tensor from the inference process.
    """
    if input_type == "raw":
        return data

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
            output = model(input_ids)[0]

            if input_type == "cls_emb":
                batch_embeddings = output[:, 0, :]
            elif input_type == "channel_emb":
                batch_embeddings = output[:, 1:, :]
            else:
                raise ValueError(f"Invalid input_type: '{input_type}'. Must be 'cls_emb' or 'channel_emb'.")
            
            embeddings.append(batch_embeddings)
            
    return torch.cat(embeddings, dim=0).float()
