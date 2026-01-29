# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:23:54 2024

This script defines the LWM model architecture.

@author: Sadjad Alikhani
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class Embedding(nn.Module):
    def __init__(self, element_length, d_model, max_len=513):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)  
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        seq_len = x.size(1) 
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device) 
        pos_encodings = self.pos_embed(pos)  
        tok_emb = self.proj(x.float()) 
        embedding = tok_emb + pos_encodings 
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = self.scaled_dot_attn(q_s, k_s, v_s)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(output)
        return residual + self.dropout(output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, enc_inputs):
        # Self-Attention with Add & Norm
        attn_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        attn_outputs = self.norm1(enc_inputs + attn_outputs)  # Add & Norm

        # Feed-Forward with Add & Norm
        ff_outputs = self.pos_ffn(attn_outputs)
        enc_outputs = self.norm2(attn_outputs + ff_outputs)  # Add & Norm

        return enc_outputs, attn


class lwm(nn.Module):
    def __init__(self, element_length=32, d_model=128, n_layers=12, max_len=513, n_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = Embedding(element_length, d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_model*4, dropout) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNormalization(d_model)

        embed_weight = self.embedding.proj.weight
        _, n_dim = embed_weight.size()
        self.decoder = nn.Linear(d_model, n_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_dim))

    @classmethod
    def from_pretrained(cls, ckpt_name='model_weights.pth', device='cuda'):
        model = cls().to(device)
        state_dict = torch.load(ckpt_name, map_location=device)
        
        # Robust Key Cleaning
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
            
        missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
        
        if missing or unexpected:
            print(f"Model loaded. Missing keys: {missing}, Unexpected: {unexpected}")
            
        return model

    def forward(self, input_ids, masked_pos=None):
        # Step 1: Embedding
        output = self.embedding(input_ids)
        attention_maps = []

        # Step 2: Pass through Encoder Layers
        for layer in self.layers:
            output, attn = layer(output)
            attention_maps.append(attn)

        # If masked_pos is provided, perform masked token prediction
        if masked_pos is not None:
            masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(F.relu(self.linear(h_masked))) 
            logits_lm = self.decoder(h_masked) + self.decoder_bias
            return logits_lm, output, attention_maps
        else:
            return output, attention_maps

"""
This script defined below was created by me.

@author: Murilo Ferreira Alves Batista
"""

class Tokenizer:
    """
    A Tokenizer that generates tokens for the LWM from wireless channels

    Attributes:
        patch_rows (int): The number of rows used in each patch
        patch_cols (int): The number of columns used in each patch
        cls_value (float): The value that represents the CLS token
        scale_factor (int): The scale factor for normalization
    """
    def __init__(self,
                 patch_rows: int,
                 patch_cols: int,
                 cls_value: float = 0.2,
                 scale_factor: int = 1e6):
        """
        Constrcutor of the tokenizer

        Inputs:
            patch_rows (int): The number of rows used in each patch
            patch_cols (int): The number of columns used in each patch
            cls_value (float): The value that represents the CLS token
            scale_factor (int): The scale factor for normalization
        """
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.cls_value = cls_value
        self.scale_factor = scale_factor

    def patching(self,
                 x_complex: torch.Tensor) -> torch.Tensor:
        """
        Generates patches from the complex channels
        
        Inputs:
            x_complex (torch.Tensor): The complex wireless channel [B, M, N, SC] or [B, N, SC]

        Outputs:
            patches (torch.Tensor): The final patches [B, Patches, Features]
        """
        # Step 1: Dimension check
        # Remove Singleton dimension - Currently, the LWM model uses only one antenna in the UE
        if x_complex.ndim == 4:
            x_complex = x_complex[:, 0, :, :]

        batch_size, n_rows, n_cols = x_complex.shape

        # Step 2: Split into real and imaginary parts and interleave them
        x_real = x_complex.real
        x_imag = x_complex.imag
        x_interleaved = torch.stack((x_real, x_imag), dim=-1).flatten(start_dim=2)

        # 3. Calculate Padding
        current_rows = x_interleaved.shape[1]
        current_cols = x_interleaved.shape[2]
        patch_width_flat = self.patch_cols * 2 # Real+Imag width

        pad_rows = (self.patch_rows - (current_rows % self.patch_rows)) % self.patch_rows
        pad_cols = (patch_width_flat - (current_cols % patch_width_flat)) % patch_width_flat

        # 4. Apply Padding
        if pad_rows > 0 or pad_cols > 0:
            x_interleaved = F.pad(x_interleaved, (0, pad_cols, 0, pad_rows), value=0)

        # 5. Unfold (Create Patches)
        # Shape: (B, n_pr, Width, h)
        patches = x_interleaved.unfold(dimension=1, size=self.patch_rows, step=self.patch_rows)
        # Shape: (B, n_pr, n_pc, h, w)
        patches = patches.unfold(dimension=2, size=patch_width_flat, step=patch_width_flat)

        # 6. Flatten to Sequence
        # Permute to (B, n_pr, n_pc, h, w)
        # We need to keep this permutation logic consistent for folding back
        patches = patches.contiguous()

        # 7. Flatten grid (n_pr, n_pc) into Num_Patches
        patches = patches.flatten(start_dim=1, end_dim=2)

        # 8. Flatten pixels (h, w) into Features
        patches = patches.flatten(start_dim=2)

        return patches
    
    def tokenizing(self,
                   patches: torch.Tensor) -> torch.Tensor:
        """
        Generates tokens used in the LWM from tokens.
        Basically, prepends a CLS token in the beginning of the token sequence

        Inputs:
            patches (torch.Tensor): The patches used to produce tokens [B, Patches, Features]

        Outputs:
            tokens (torch.Tensor): The sequence of tokens [B, Sequence Length, Features]
        """
        batch_size = patches.shape[0]
        features = patches.shape[-1]
        device = patches.device

        # 1. Create CLS token batch
        cls_tokens = torch.full(
            (batch_size, 1, features), 
            self.cls_value, 
            device=device, 
            dtype=patches.dtype
        )

        # 2. Prepend CLS
        return torch.cat([cls_tokens, patches], dim=1)
    
    def __call__(self,
                 x_complex: torch.Tensor) -> torch.Tensor:
        """
        Transform the complex wireless channel into a sequence of tokens and multiplies for normalization.
        
        Inputs:
            x_complex (torch.Tensor): The complex wireless channel [B, K, N, SC] or [B, N, SC]

        Outputs:
            tokens (torch.Tensor): The sequence of tokens [B, K, Sequence Length, Features] or [B, Sequence Length, Features]
        """
        input_ndim = x_complex.ndim 
        x_complex = x_complex * self.scale_factor # Scale factor for LWM
        # 1. Handle dimensions
        if input_ndim == 4:
            # Case A: (Batch, Users, M, S)
            batch_dim, user_dim, n_rows, n_cols = x_complex.shape
            # Flatten Batch and User together for processing
            # New shape: (B*Users, M, S)
            x_processing = x_complex.reshape(-1, n_rows, n_cols)

        elif input_ndim == 3:
            # Case B: (Batch, M, S)
            x_processing = x_complex 
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x_complex.shape}")
        # 2. Process
        patches = self.patching(x_processing)
        tokens = self.tokenizing(patches)
        # 3. Restore original shape
        if input_ndim == 4:
            # Un-flatten (Batch*Users) -> (Batch, Users)
            # Current: (Batch*Users, Sequence, Dim)
            # New shape: (Batch, Users, Sequence, Dim)
            seq_len = tokens.shape[1]
            token_dim = tokens.shape[2]

            tokens = tokens.view(batch_dim, user_dim, seq_len, token_dim)

        return tokens

class LWMWrapper(nn.Module):
    def __init__(self, tokenizer, lwm_model, task_head, mode="cls"):
        super().__init__()
        self.tokenizer = tokenizer
        self.lwm = lwm_model
        self.task_head = task_head
        self.mode = mode

        for param in self.lwm.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, fine_tune_layers):
        """
        Selectively unfreezes layers in the LWM backbone.
        
        Args:
            fine_tune_layers: "full" (all layers) OR list of strings (e.g. ["layers.10", "layers.11"])
        """
        # We target self.lwm for unfreezing
        model_ref = self.lwm
        
        if fine_tune_layers == "full":
            print("Unfreezing all LWM layers.")
            for param in model_ref.parameters():
                param.requires_grad = True
        else:
            # Get available layer names for validation
            available_layers = [name for name, _ in model_ref.named_parameters()]
            
            # Validate requests
            for layer_req in fine_tune_layers:
                if not any(layer_req in lname for lname in available_layers):
                    raise ValueError(
                        f"Layer substring '{layer_req}' not found in LWM model.\n"
                        f"Example available layers: {available_layers[:3]}..."
                    )
            
            # Perform selective unfreezing
            print(f"Unfreezing LWM layers matching: {fine_tune_layers}")
            for name, param in model_ref.named_parameters():
                if any(layer_req in name for layer_req in fine_tune_layers):
                    param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: Can be (Batch, 32, 32) OR (Batch, K_Users, 32, 32)
        Returns:
            (Batch, Dim) OR (Batch, K_Users, Dim)
        """
        # 1. Shape handling
        if x.ndim == 4: # Input is (Batch, K, 32, 32)
            B, K, M, N = x.shape
            # Flatten Batch and K to process in parallel: (B*K, 32, 32)
            tokenizer_input = x.view(B*K, M, N)
        else:
            tokenizer_input = x

        # 2. Tokenizer
        # (Total_Batch, Seq, element_length)
        tokens = self.tokenizer(tokenizer_input)
        
        # 3. LWM forward pass
        # embeddings: (Total_Batch, Seq_Len, d_model)
        embeddings, _ = self.lwm(tokens)
        
        # 4. Extract features
        if self.mode == "cls":
            # Take CLS token: (Total_Batch, d_model)
            features = embeddings[:, 0, :]
        elif self.mode == "channel_emb":
            # Take Channel tokens: (Total_Batch, (Seq_Len-CLS)*d_model)
            features = embeddings[:, 1:, :].flatten(start_dim=1)
        elif self.mode == "combined":
            # All tokens: (Total_Batch, Seq_Len*d_model)
            features = embeddings.flatten(start_dim=1)
        elif self.mode == "mean_pooled":
            # Mean pooled: (Total_Batch, Dim)
            features = torch.mean(embeddings, dim=1).unsqueeze(1) 
        else:
            raise ValueError(f"Invalid LWM mode: {self.mode}")

        # 5. Restore multi-user shape
        if x.ndim == 4:
           # (B, K, Dim)
            # features = features.view(B, K, -1)
            features = features.reshape(B, -1)

        out = self.task_head(features)
            
        return out