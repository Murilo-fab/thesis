# -*- coding: utf-8 -*-
"""
LWM Model Architecture and Tokenization.

This file contains two distinct sections:
1. The LWM Transformer Architecture (Adapted from Sadjad Alikhani).
2. The Wireless Channel Tokenizer (Created by Murilo Ferreira Alves Batista).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# PART 1: LWM MODEL ARCHITECTURE
# Created on Fri Sep 13 19:23:54 2024
# @author: Sadjad Alikhani
# =============================================================================

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

# =============================================================================
# PART 2: WIRELESS CHANNEL TOKENIZER
# =============================================================================

"""
The script/class defined below was created by Murilo Ferreira Alves Batista
"""

# ... (Previous imports: torch, nn, F, np, Optional) ...

class Tokenizer:
    """
    Transforms complex wireless channel matrices into token sequences for the LWM.
    
    Supports:
    1. Standard Tokenization: [CLS, Patches]
    2. Masked Language Modeling (MLM): Randomly masks patches for pre-training.

    Attributes:
        patch_rows (int): Height of each patch.
        patch_cols (int): Width of each patch.
        cls_value (float): Value for the [CLS] token.
        mask_value (float): Value for the [MASK] token.
        max_len (int): Maximum allowed sequence length.
    """
    def __init__(self, 
                 patch_rows: int, 
                 patch_cols: int, 
                 cls_value: float = 0.2, 
                 mask_value: float = 0.1,
                 max_len: int = 513):
        
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.cls_value = cls_value
        self.mask_value = mask_value
        self.max_len = max_len

    def patching(self, x_complex: torch.Tensor) -> torch.Tensor:
        """Splits complex channel into flattened patches."""
        # 1. Dimension Standardization
        if x_complex.ndim == 4:
            if x_complex.size(1) == 1:
                x_complex = x_complex.squeeze(1)
            else:
                x_complex = x_complex[:, 0, :, :]

        batch_size, n_rows, n_cols = x_complex.shape

        # 2. Interleave Real/Imag
        x_real = x_complex.real
        x_imag = x_complex.imag
        x_interleaved = torch.stack((x_real, x_imag), dim=-1).flatten(start_dim=2)

        # 3. Padding
        patch_feature_width = self.patch_cols * 2 
        current_rows = x_interleaved.shape[1]
        current_cols = x_interleaved.shape[2]

        pad_rows = (self.patch_rows - (current_rows % self.patch_rows)) % self.patch_rows
        pad_cols = (patch_feature_width - (current_cols % patch_feature_width)) % patch_feature_width

        if pad_rows > 0 or pad_cols > 0:
            x_interleaved = F.pad(x_interleaved, (0, pad_cols, 0, pad_rows), value=0)

        # 4. Unfold & Flatten
        patches = x_interleaved.unfold(1, self.patch_rows, self.patch_rows)
        patches = patches.unfold(2, patch_feature_width, patch_feature_width)
        patches = patches.contiguous().flatten(start_dim=1, end_dim=2).flatten(start_dim=2)

        return patches

    def apply_masking(self, patches: torch.Tensor, masking_percent: float = 0.40):
        """
        Applies BERT-style masking to the patches.
        
        Args:
            patches (Tensor): [Batch, Num_Patches, Features]
            masking_percent (float): Fraction of tokens to mask.
            
        Returns:
            masked_patches (Tensor): Patches with some replaced by [MASK].
            target_patches (Tensor): The original patches that were masked (Ground Truth).
            masked_indices (Tensor): Indices of masked tokens (adjusted for CLS offset).
        """
        batch_size, num_patches, features = patches.shape
        device = patches.device
        
        # Determine number of tokens to mask
        n_mask = int(masking_percent * num_patches)
        
        # Create mask token
        mask_token = torch.full((1, 1, features), self.mask_value, device=device)

        # 1. Select indices to mask
        # We start from index 1 because index 0 will be CLS
        # Generate random permutation for every sample in batch
        rand_indices = torch.rand(batch_size, num_patches, device=device).argsort(dim=1)
        mask_idx = rand_indices[:, :n_mask]  # [B, n_mask]

        # 2. Create copies for output
        masked_patches = patches.clone()
        target_patches = torch.zeros(batch_size, n_mask, features, device=device)

        # 3. Apply Masking
        # Gather targets
        for b in range(batch_size):
            indices = mask_idx[b]
            target_patches[b] = patches[b, indices, :]
            masked_patches[b, indices, :] = mask_token.squeeze()

        # Adjust indices by +1 because CLS token will be prepended later
        mask_pos = mask_idx + 1

        return masked_patches, target_patches, mask_pos

    def tokenizing(self, patches: torch.Tensor) -> torch.Tensor:
        """Prepends [CLS] token."""
        batch_size, num_patches, features = patches.shape
        
        if num_patches + 1 > self.max_len:
            raise ValueError(f"Sequence length {num_patches + 1} exceeds max_len {self.max_len}")

        cls_tokens = torch.full(
            (batch_size, 1, features), 
            self.cls_value, 
            device=patches.device, 
            dtype=patches.dtype
        )
        return torch.cat([cls_tokens, patches], dim=1)

    def __call__(self, x_complex: torch.Tensor, mask: bool = False, masking_percent: float = 0.40):
        """
        Args:
            x_complex: Input channel matrix.
            mask (bool): If True, returns masked inputs and targets for pre-training.
            masking_percent (float): Percentage of tokens to mask.
            
        Returns:
            If mask=False: tokens [Batch, Seq_Len, Dim]
            If mask=True:  (tokens, target_patches, masked_indices)
        """
        input_ndim = x_complex.ndim

        # Handle batching dimensions
        if input_ndim == 4:
            batch_dim, user_dim, n_rows, n_cols = x_complex.shape
            x_processing = x_complex.reshape(-1, n_rows, n_cols)
        elif input_ndim == 3:
            x_processing = x_complex
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x_complex.shape}")

        # 1. Patching
        patches = self.patching(x_processing)

        # 2. Masking (Optional)
        target_patches = None
        masked_indices = None
        
        if mask:
            patches, target_patches, masked_indices = self.apply_masking(patches, masking_percent)

        # 3. Tokenizing (Add CLS)
        tokens = self.tokenizing(patches)

        # 4. Reshape Restore
        if input_ndim == 4:
            seq_len = tokens.shape[1]
            token_dim = tokens.shape[2]
            tokens = tokens.view(batch_dim, user_dim, seq_len, token_dim)
            
            # Note: If masking is ON with 4D input, targets/indices will also need reshaping 
            # if you intend to use them per-user. Typically pre-training uses flattened batches.

        if mask:
            return tokens, target_patches, masked_indices
        return tokens
    
"""
LWM Pre-training Script (Masked Channel Modeling).

This script performs self-supervised pre-training of the Large Wireless Model (LWM).
It uses a "Bucketing" strategy to handle variable sequence lengths (caused by different 
patching dimensions or scenarios) by creating separate DataLoaders for each sequence length.

Key Components:
1. Masked Token Prediction (BERT-style).
2. NMSE Loss on masked patches.
3. Warmup + Cosine Decay Scheduler.

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

import os
import csv
import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Local Imports
import DeepMIMOv3
from thesis.lwm_model import lwm, Tokenizer
from thesis.utils import get_parameters, count_parameters

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    """Hyperparameters for LWM Pre-training."""
    # Identity
    TASK_NAME: str = "lwm_pretrain_masked"
    
    # Data Generation
    TRAIN_SCENARIOS: List[str] = field(default_factory=lambda: [
        "city_7_sandiego", "city_11_santaclara", "city_12_fortworth", 
        "city_15_indianapolis", "city_19_oklahoma"
    ])
    VAL_SCENARIO: str = "city_18_denver"
    
    # Tokenizer settings
    PATCH_ROWS: int = 4
    PATCH_COLS: int = 4
    SCALE_FACTOR: float = 1e6
    MASK_PERCENT: float = 0.40
    
    # Model Architecture
    ELEMENT_LENGTH: int = 32 # (4*4*2)
    D_MODEL: int = 128
    N_LAYERS: int = 12
    N_HEADS: int = 8
    MAX_LEN: int = 513
    DROPOUT: float = 0.1
    
    # Training
    EPOCHS: int = 50
    BATCH_SIZE: int = 128
    VAL_BATCH_SIZE: int = 64
    BASE_LR: float = 5e-4
    MIN_LR: float = 1e-8
    WARMUP_EPOCHS: int = 5
    WEIGHT_DECAY: float = 0.05
    
    # Hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42
    
    # Paths
    SAVE_DIR: str = "./models/pretrain"

# -----------------------------------------------------------------------------
# DATA GENERATION (BUCKETING)
# -----------------------------------------------------------------------------

class PretrainDataGenerator:
    """
    Generates masked data from DeepMIMO and organizes it into buckets based on sequence length.
    """
    def __init__(self, config: PretrainConfig):
        self.cfg = config
        self.tokenizer = Tokenizer(
            patch_rows=config.PATCH_ROWS,
            patch_cols=config.PATCH_COLS,
            scale_factor=config.SCALE_FACTOR,
            max_len=config.MAX_LEN
        )

    def process_scenarios(self, scenarios: List[str]) -> Dict[int, TensorDataset]:
        """
        Loads scenarios, applies masking, and groups by sequence length.
        Returns: Dict {seq_len: TensorDataset}
        """
        grouped_data = defaultdict(list)
        total_samples = 0
        
        print(f"Generating data for {len(scenarios)} scenarios...")
        
        for scenario in scenarios:
            try:
                # 1. Load Raw Data
                params = get_parameters(scenario)
                deepmimo_data = DeepMIMOv3.generate_data(params)
                
                # Filter valid users
                los = deepmimo_data[0]['user']['LoS']
                valid = np.where(los != -1)[0]
                raw_chs = deepmimo_data[0]['user']['channel'][valid] # (N, 1, Tx, SC)
                
                if raw_chs.ndim == 4:
                    raw_chs = raw_chs.squeeze(axis=1) # (N, Tx, SC)
                
                # 2. Tokenize & Mask
                # Convert to Tensor
                x_complex = torch.tensor(raw_chs, dtype=torch.complex64)
                
                # Apply Tokenizer with Masking enabled
                # Returns: (tokens, targets, mask_indices)
                tokens, targets, mask_pos = self.tokenizer(
                    x_complex, 
                    mask=True, 
                    masking_percent=self.cfg.MASK_PERCENT
                )
                
                # 3. Group by Sequence Length
                # tokens shape: [Batch, Seq_Len, Feat]
                seq_len = tokens.shape[1]
                
                # Store tuple (tokens, targets, mask_pos)
                grouped_data[seq_len].append((tokens, targets, mask_pos))
                total_samples += tokens.shape[0]
                
            except Exception as e:
                print(f"Skipping {scenario}: {e}")

        # 4. Consolidate into TensorDatasets
        bucketed_datasets = {}
        print(f"Consolidating buckets...")
        
        for length, data_list in grouped_data.items():
            # data_list is a list of tuples [(tok, tar, pos), (tok, tar, pos)...]
            # Unzip them
            all_tokens = torch.cat([x[0] for x in data_list], dim=0)
            all_targets = torch.cat([x[1] for x in data_list], dim=0)
            all_pos = torch.cat([x[2] for x in data_list], dim=0)
            
            bucketed_datasets[length] = TensorDataset(all_tokens, all_targets, all_pos)
            
        print(f"Total Samples: {total_samples} | Buckets: {list(bucketed_datasets.keys())}")
        return bucketed_datasets

# -----------------------------------------------------------------------------
# LOSS & TRAINING
# -----------------------------------------------------------------------------

def nmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates NMSE between predicted masked tokens and ground truth.
    y_pred: [Batch, N_Masked, Feat]
    y_true: [Batch, N_Masked, Feat]
    """
    # Flatten features
    y_pred_flat = y_pred.reshape(y_pred.size(0), -1)
    y_true_flat = y_true.reshape(y_true.size(0), -1)

    mse = torch.sum((y_true_flat - y_pred_flat)**2, dim=-1)
    power = torch.sum(y_true_flat**2, dim=-1)
    
    return mse / (power + 1e-8)

def train_lwm_epoch(model, loaders, optimizer, scheduler, device):
    """Runs one training epoch over all buckets."""
    model.train()
    total_nmse = 0.0
    total_samples = 0
    
    # Iterate over buckets (Sequence Lengths)
    for length, loader in loaders.items():
        with tqdm(loader, desc=f"Train Len {length}", unit="batch", leave=False) as t:
            for batch in t:
                # Unpack: Input Tokens (with [MASK]), Target Patches, Mask Indices
                input_ids, targets, mask_pos = [b.to(device) for b in batch]
                
                optimizer.zero_grad()
                
                # Forward Pass
                # LWM forward needs mask_pos to select which tokens to predict
                logits_lm, _, _ = model(input_ids, mask_pos)
                
                # Loss Calculation
                loss_batch = torch.sum(nmse_loss(logits_lm, targets))
                loss_mean = loss_batch / input_ids.size(0) # For optimization scaling
                
                loss_mean.backward()
                optimizer.step()
                scheduler.step()
                
                # Metrics
                batch_size = input_ids.size(0)
                total_nmse += loss_batch.item()
                total_samples += batch_size
                
                t.set_postfix({"NMSE": total_nmse/total_samples, "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
                
    return total_nmse / max(total_samples, 1)

def validate_lwm(model, loaders, device):
    """Runs validation."""
    model.eval()
    total_nmse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for length, loader in loaders.items():
            for batch in loader:
                input_ids, targets, mask_pos = [b.to(device) for b in batch]
                
                logits_lm, _, _ = model(input_ids, mask_pos)
                
                loss_sum = torch.sum(nmse_loss(logits_lm, targets))
                total_nmse += loss_sum.item()
                total_samples += input_ids.size(0)
                
    return total_nmse / max(total_samples, 1)

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PretrainConfig()
    
    # 1. Setup
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(cfg.SAVE_DIR, start_time)
    os.makedirs(run_dir, exist_ok=True)
    
    log_path = os.path.join(run_dir, "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(["Epoch", "Train_NMSE", "Val_NMSE", "LR"])

    # 2. Data Preparation
    generator = PretrainDataGenerator(cfg)
    
    # Load separate buckets
    train_buckets = generator.process_scenarios(cfg.TRAIN_SCENARIOS)
    val_buckets = generator.process_scenarios([cfg.VAL_SCENARIO])
    
    # Create DataLoaders
    train_loaders = {k: DataLoader(v, batch_size=cfg.BATCH_SIZE, shuffle=True) for k, v in train_buckets.items()}
    val_loaders = {k: DataLoader(v, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False) for k, v in val_buckets.items()}

    # 3. Model
    model = lwm(
        element_length=cfg.ELEMENT_LENGTH,
        d_model=cfg.D_MODEL,
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEADS,
        max_len=cfg.MAX_LEN,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)
    
    print(f"Model Parameters: {count_parameters(model):,}")

    # 4. Optimizer & Scheduler
    # Calculate total steps for correct cosine decay
    total_steps_epoch = sum(len(l) for l in train_loaders.values())
    total_steps = total_steps_epoch * cfg.EPOCHS
    warmup_steps = total_steps_epoch * cfg.WARMUP_EPOCHS
    
    optimizer = AdamW(model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999))
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1 - cfg.MIN_LR/cfg.BASE_LR) + cfg.MIN_LR/cfg.BASE_LR

    scheduler = LambdaLR(optimizer, lr_lambda)

    # 5. Training Loop
    best_nmse = float('inf')
    
    print(f"\nStarting Pre-training for {cfg.EPOCHS} epochs...")
    
    for epoch in range(cfg.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{cfg.EPOCHS} ---")
        
        # Train
        train_nmse = train_lwm_epoch(model, train_loaders, optimizer, scheduler, cfg.DEVICE)
        
        # Validate
        val_nmse = validate_lwm(model, val_loaders, cfg.DEVICE)
        
        # Log
        lr_curr = scheduler.get_last_lr()[0]
        print(f"Train NMSE: {train_nmse:.6f} | Val NMSE: {val_nmse:.6f} | LR: {lr_curr:.2e}")
        
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, train_nmse, val_nmse, lr_curr])
            
        # Save Best
        if val_nmse < best_nmse:
            best_nmse = val_nmse
            save_path = os.path.join(run_dir, f"lwm_best_mask{cfg.MASK_PERCENT}_nmse{val_nmse:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f" >> Model Saved: {save_path}")

    print("\nPre-training Complete.")