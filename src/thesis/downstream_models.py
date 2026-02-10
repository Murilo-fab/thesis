"""
Downstream Model Architectures.

This module defines:
1. Task Heads: Classification (Linear), Regression (Softmax), and Enhanced Heads.
2. Baselines: 1D-CNN (ResNet-style) for comparison.
3. Wrapper: A unified interface to connect Tokenizers/Encoders (LWM/AE) with Task Heads.
4. Factory: A builder function to instantiate models from config files.

Author: Murilo Ferreira Alves Batista - RWTH Aachen/USP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local Imports
from thesis.csi_autoencoder import CSIAutoEncoder
from thesis.lwm_model import lwm, Tokenizer 

# =============================================================================
# PART 1: TASK HEADS
# =============================================================================

class ClassificationHead(nn.Module):
    """
    Standard MLP for Classification (Beam Prediction, LoS/NLoS).
    Structure: Linear -> BN -> ReLU -> Dropout -> ...
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [Batch, Features] or [Batch, Seq, Emb] -> Flatten
        x = x.flatten(start_dim=1)
        return self.classifier(x)

class EnhancedClassificationHead(nn.Module):
    """
    Classification Head with Logarithmic Preprocessing.
    
    Physics Note:
    Wireless channel features often span orders of magnitude. 
    Applying log10(abs(x)) converts values to dB-scale, which is easier for MLPs to learn.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        # Log-scale transformation (dB-like)
        x = torch.log10(torch.abs(x) + 1e-9)
        return self.classifier(x)

class RegressionHead(nn.Module):
    """
    Regression Head for Power Allocation.
    Output is Softmax-normalized to ensure Sum(Power) = 1 constraint.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.ReLU(),                
            nn.Linear(256, output_dim),
            nn.Softmax(dim=1) # Enforce Power Budget Constraint
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.regressor(x)

# =============================================================================
# PART 2: BASELINE MODELS (1D CNN)
# =============================================================================

class ResidualBlock(nn.Module):
    """ Standard 1D Residual Block for Res1DCNN. """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class Res1DCNN(nn.Module):
    """
    Universal Downstream Baseline: 1D ResNet-style CNN.
    Used to benchmark if deep learning adds value over standard CNNs.
    """
    def __init__(self, input_channels, num_classes):
        super(Res1DCNN, self).__init__()
        self.in_ch = input_channels

        self.features = nn.Sequential(
            # Stem
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Residual Layers (Depths: 2, 3, 4)
            self._make_layer(32, 32, 2),
            self._make_layer(32, 64, 3),
            self._make_layer(64, 128, 4),
            
            # Global Pooling
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Aliases for unified interface
        self.encoder = self.features
        self.task_head = self.classifier

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = [ResidualBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [Batch, Len, Channels] -> [Batch, Channels, Len]
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

# =============================================================================
# PART 3: UNIFIED WRAPPER
# =============================================================================

class Wrapper(nn.Module):
    """
    Universal Adapter that connects:
    Input -> [Tokenizer] -> [Encoder] -> [Task Head] -> Output
    
    Handles shape mismatches between Single-User (3D) and Multi-User (4D) data.
    """
    def __init__(self, tokenizer=None, encoder=None, task_head=None, mode="raw"):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.task_head = task_head
        self.mode = mode

    def get_features(self, x):
        """
        Extracts latent representations.
        
        Input Support:
          - Single-User: (Batch, Tx, SC)
          - Multi-User:  (Batch, K_Users, Tx, SC)
        
        Logic:
          1. Detects if input is Multi-User (4D).
          2. Flattens Multi-User batch dimension: (B, K, ...) -> (B*K, ...).
          3. Passes through Encoder (LWM or AE).
          4. Reshapes back to (B, K, Features) if needed.
        """
        # 1. Detect Input Mode
        is_multiuser = (x.ndim == 4)
        
        # 2. Conditional Folding
        if is_multiuser:
            B, K = x.shape[0], x.shape[1]
            x_proc = x.flatten(0, 1) # Merge Batch & Users
        else:
            x_proc = x

        # Path A: Baseline (Raw Complex Features)
        if self.mode == "raw":
            # Concat Real/Imag channel-wise
            return torch.cat((x.real, x.imag), dim=-1)

        # Path B: AutoEncoder (AE)
        elif self.mode == "ae":
            if not self.encoder:
                raise ValueError("Mode 'ae' requires an encoder.")
            
            # Forward pass through AE Encoder
            feats = self.encoder(x_proc)
            
            # Reshape back
            if is_multiuser:
                return feats.view(B, K, -1)
            else:
                return feats.unsqueeze(1) # Add sequence dim
        
        # Path C: LWM (Tokenizer + Transformer)
        elif self.tokenizer:
            # 1. Tokenize (Scale -> Patch -> Add CLS)
            tokens = self.tokenizer(x_proc)
            
            # 2. Transformer Encoder
            # Returns: (Batch, Seq_Len, Emb_Dim), Attention_Maps
            embeddings, _ = self.encoder(tokens)
            
            # Mode C1: CLS Token Only (Classification)
            if self.mode == "cls":
                out = embeddings[:, 0:1, :] # Take index 0
                
                if is_multiuser:
                    return out.view(B, K, -1)
                else:
                    return out

            # Mode C2: Full Channel Embeddings (Power Allocation)
            if self.mode == "channel_emb":
                out = embeddings[:, 1:, :] # Skip CLS, keep patches
                
                if is_multiuser:
                    # (B, K, Seq_Len, Emb_Dim)
                    return out.view(B, K, out.shape[1], -1)
                else:
                    return out
            
        raise ValueError(f"Incompatible mode '{self.mode}' configuration.")

    def forward(self, x):
        features = self.get_features(x)
        return self.task_head(features)

# =============================================================================
# PART 4: FACTORY & UTILS
# =============================================================================

def build_model_from_config(config):
    """
    Factory function: Instantiates a complete model pipeline from config.
    """
    # 1. Build Task Head
    head_map = {
        "classification": ClassificationHead,
        "enhanced_classification": EnhancedClassificationHead,
        "regression": RegressionHead,
        "residual_1d_cnn": Res1DCNN
    }
    
    if config.task_type not in head_map:
        raise ValueError(f"Unknown task_type: {config.task_type}")
        
    task_head = head_map[config.task_type](config.input_size, config.output_size)

    # 2. Build Backbone
    tokenizer, encoder = None, None

    # Case A: Autoencoder
    if config.encoder_type == "AE":
        if not config.weights_path:
            print("Warning: Initializing AE with random weights (No path provided).")
        encoder = CSIAutoEncoder(latent_dim=config.latent_dim)
        if config.weights_path:
            encoder.load_weights(config.weights_path)
            
    # Case B: LWM
    elif config.encoder_type == "LWM":
        # Standard Tokenizer setup for LWM
        tokenizer = Tokenizer(patch_rows=4, patch_cols=4, scale_factor=1e0)
        encoder = lwm.from_pretrained(config.weights_path)

    # 3. Fine-tuning Control (Freeze/Unfreeze)
    if encoder:
        # Default: Freeze everything
        for param in encoder.parameters():
            param.requires_grad = False
            
        # Unfreeze specific layers if requested
        unfreeze_layers(encoder, config.fine_tune_layers)

    return Wrapper(tokenizer, encoder, task_head, config.mode)

def unfreeze_layers(model, fine_tune_layers):
    """Helper to unfreeze specific layers based on substring matching."""
    if not fine_tune_layers or not model:
        return
    
    if fine_tune_layers == "full":
        print(f"Unfreezing ALL layers in {model.__class__.__name__}")
        for param in model.parameters():
            param.requires_grad = True
        return
    
    # Check if requested layers actually exist
    available_layers = [name for name, _ in model.named_parameters()]
    
    print(f"Unfreezing layers matching: {fine_tune_layers}")
    for layer_req in fine_tune_layers:
        if not any(layer_req in lname for lname in available_layers):
            print(f"Warning: Layer substring '{layer_req}' not found in model.")
        
        for name, param in model.named_parameters():
            if layer_req in name:
                param.requires_grad = True
