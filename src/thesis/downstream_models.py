import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.csi_autoencoder import CSIAutoEncoder
from thesis.lwm_model import lwm, Tokenizer 

class ClassificationHead(nn.Module):
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
        return self.classifier(x)
    
class EnhancedClassificationHead(nn.Module):
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
        x = torch.log10(torch.abs(x) + 1e-9)
        return self.classifier(x)
    
class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim):

        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.BatchNorm1d(512),      
            nn.ReLU(),                
            nn.Dropout(0.1),          
            nn.Linear(512, 256),     
            nn.BatchNorm1d(256),     
            nn.ReLU(),                 
            nn.Dropout(0.1),          
            nn.Linear(256, output_dim)      
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.regressor(x)
    
class ResidualBlock(nn.Module):
    """ Standard 1D Residual Block from LWM Benchmark """
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
    Universal Downstream 1D-CNN
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
        
        self.encoder = self.features
        self.task_head = self.classifier

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = [ResidualBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class Wrapper(nn.Module):
    def __init__(self, tokenizer=None, encoder=None, task_head=None, mode="raw"):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.task_head = task_head
        self.mode = mode

    def get_features(self, x):
        # 1. Baseline: No encoding
        if self.mode == "raw":
            return torch.cat((x.real, x.imag), dim=-1)
        
        # 2. AutoEncoder Path
        if self.mode == "ae":
            if not self.encoder:
                raise ValueError("Mode 'ae' requires an encoder.")
            feats = self.encoder(x)
            feats = feats[0] if isinstance(feats, (tuple, list)) else feats
            return feats.unsqueeze(1)
        
        # 3. LWM
        if self.tokenizer:
            tokens = self.tokenizer(x)
            embeddings, _ = self.encoder(tokens)
            
            if self.mode == "cls":
                return embeddings[:, 0:1, :]
            if self.mode == "channel_emb":
                return embeddings[:, 1:, :]
            
        raise ValueError(f"Incompatible mode '{self.mode}' and components.")

    def forward(self, x):
        features = self.get_features(x)
        return self.task_head(features)

def build_model_from_config(config):
    """
    Factory function that instantiates a model based on the provided configuration.
    """
    # 1. Build task head
    head_map = {"classification": ClassificationHead,
                "enhanced_classification": EnhancedClassificationHead,
                "regression": RegressionHead,
                "residual_1d_cnn": Res1DCNN}
    if config.task_type not in head_map:
        return None
    task_head = head_map[config.task_type](config.input_size, config.output_size)

    # 2. Backbone selection
    tokenizer, encoder = None, None

    # Case A: Autoencoder
    if config.encoder_type == "AE":
        encoder = CSIAutoEncoder(latent_dim=config.latent_dim)
        encoder.load_weights(config.weights_path)
    # Case B: LWM
    elif config.encoder_type == "LWM":
        tokenizer = Tokenizer(patch_rows=4, patch_cols=4, scale_factor=1e0)
        encoder = lwm.from_pretrained(config.weights_path)

    # 3. Grad Management
    if encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        unfreeze_layers(encoder, config.fine_tune_layers)

    return Wrapper(tokenizer, encoder, task_head, config.mode)

def unfreeze_layers(model, fine_tune_layers):
    if not fine_tune_layers or not model:
        return
    
    if fine_tune_layers == "full":
        for param in model.parameters():
            param.requires_grad = True
        return
    
    available_layers = [name for name, _ in model.named_parameters()]

    for layer_req in fine_tune_layers:
        if not any(layer_req in lname for lname in available_layers):
            raise ValueError(f"Layer substring '{layer_req}' not found model.")
        
        for name, param in model.named_parameters():
            if any(layer_req in name for layer_req in fine_tune_layers):
                param.requires_grad = True