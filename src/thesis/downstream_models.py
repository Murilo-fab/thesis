import torch
import torch.nn as nn
from torch.functional import F

class PowerAllocator(nn.Module):
    def __init__(self, emb_dim, num_subcarriers, hidden_dim=64):
        """
        Universal Power Allocation Head.
        - Supports Batch Normalization for training stability.
        - Automatically handles scalar (CLS), patch, or full-resolution inputs.
        """
        super().__init__()
        self.S_target = num_subcarriers
        
        # 1. The Core Network (BN-MLP)
        self.layer1 = nn.Linear(emb_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.head = nn.Linear(hidden_dim // 2, 1)
        
        # Optional: Learnable scaler allows the model to learn "sharp" waterfilling cutoffs
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward_mlp(self, x):
        """
        Helper to run MLP with BatchNorm on arbitrary input shapes.
        BN requires (N, C), so we flatten and restore dimensions safely.
        """
        # x shape: [Batch, K, Seq, emb_dim]
        original_shape = x.shape
        
        # Flatten to [N_samples, emb_dim] where N = Batch * K * Seq
        # This maximizes batch statistics for BN
        x_flat = x.reshape(-1, original_shape[-1])
        
        # Layer 1
        out = self.layer1(x_flat)
        out = self.bn1(out) 
        out = F.relu(out)
        
        # Layer 2
        out = self.layer2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # Head (No BN/ReLU on final output usually)
        out = self.head(out)
        
        # Restore shape -> [Batch, K, Seq, 1]
        return out.reshape(original_shape[:-1] + (1,))

    def forward(self, z):
        """
        Input z can be:
        1. (B, K, S_target, D) -> Full resolution (one embedding per subcarrier)
        2. (B, K, Seq, D)      -> Patch embeddings (Seq < S_target)
        3. (B, K, D)           -> CLS / Global Pooling (Seq dimension missing)
        """
        # 1. Standardization
        # Ensure input is always 4D: (B, K, Seq, D)
        if z.dim() == 3: 
            # Case: (B, K, D) -> Add sequence dim -> (B, K, 1, D)
            z = z.unsqueeze(2)
            
        B, K, Seq, D = z.shape

        # 2. Neural Network Pass (BN-MLP)
        # Input: (B, K, Seq, D) -> Output: (B, K, Seq, 1)
        scores = self.forward_mlp(z) 
        
        # Remove last dim -> (B, K, Seq)
        scores = scores.squeeze(-1)
        
        # 3. Clever Expansion (The "Resolution Adapter")
        if Seq == self.S_target:
            # Case A: Full Resolution (1-to-1 match)
            final_scores = scores
            
        elif Seq == 1:
            # Case B: Global Context (CLS Token)
            final_scores = scores.expand(-1, -1, self.S_target)
            
        else:
            # Case C: Patch Embeddings (e.g., 16 patches for 64 subcarriers)
            if self.S_target % Seq != 0:
                # Fallback: Linear Interpolation for weird sizes
                # (B*K, 1, Seq) -> (B*K, 1, S_target)
                scores_reshaped = scores.view(B*K, 1, Seq)
                final_scores = F.interpolate(scores_reshaped, size=self.S_target, mode='linear', align_corners=False)
                final_scores = final_scores.view(B, K, self.S_target)
            else:
                # Clean integer repeat (step function)
                repeat_factor = self.S_target // Seq
                final_scores = torch.repeat_interleave(scores, repeats=repeat_factor, dim=2)

        # 4. Final Formatting & Power Constraint
        # Target Shape: (Batch, Subcarriers, Users) for Sum-Rate Calc
        final_scores = final_scores.permute(0, 2, 1)
        
        # Activation (Sigmoid mapped to 0-1)
        p_raw = torch.sigmoid(final_scores * self.output_scale)
        
        # Strict Power Constraint: Sum(P) <= 1.0
        # Sum over Subcarriers (dim 1) AND Users (dim 2)
        total_power = torch.sum(p_raw, dim=(1, 2), keepdim=True)
        
        # Normalize strictly
        p_allocated = p_raw / (total_power + 1e-8)
        
        return p_allocated
    
class Wrapper(nn.Module):
    def __init__(self,
                 model,
                 task_head,
                 input_type,
                 emb_dim=128):
        super().__init__()

        self.encoder = model
        self.task_head = task_head
        self.input_type = input_type
        self.emb_dim = emb_dim

        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def fine_tune(self, fine_tune_layers="full"):
        if fine_tune_layers == "full":
             for param in self.encoder.parameters():
                    param.requires_grad = True
        else:
            for name, param in self.encoder.named_parameters():
                if any(layer in name for layer in fine_tune_layers):
                    param.requires_grad = True
         
    def forward(self, tokens):
        """
        Forward function for Wrapper
        
        Inputs:
        channels (torch.tensor): Channel matrix [B, S, K, N]

        Outputs:
        power_weights (torch.tensor): Normalized weights [B, S, K]
        """
        # 1. Extract channel shape
        B, K, S, F = tokens.shape
        
        # 2. Flatten for Encoder
        # Shape: [B, K, S, F] : [B*K, S, F]
        x = tokens.view(B*K, S, F)
        # 3. Encoder
        if self.input_type == "raw":
            input_head = tokens[:, :, 1:, :]
            input_head = input_head.view(B, K, S-1, F)
        else:
            # Embeddings shape: [B*K, S, d_model]
            embeddings, _ = self.encoder(x)
            if self.input_type == "cls":
                input_head = embeddings[:, 0, :]
                input_head = input_head.view(B, K, 1, self.emb_dim)
            elif self.input_type == "channel_emb":
                input_head = embeddings[:, 1:, :]
                input_head = input_head.view(B, K, S-1, self.emb_dim)
        # 6. Head Pass
        # The head now takes the structured data and returns normalized power
        power_weights = self.task_head(input_head)

        return power_weights