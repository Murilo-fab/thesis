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
        flat_scores = final_scores.reshape(B, -1)
        flat_probs = F.softmax(flat_scores / self.temperature, dim=-1)
        
        p_allocated = flat_probs.reshape(B, self.S_target, K)        
        # Normalize strictly
        
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
    
class AssignmentHead(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=64):
        super().__init__()

        # MLP Net
        # Input: (Batch, Users, Num_Blocks, Emb_dim)
        # Output: (Batch, User, Num_Blocks, 1)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, user_embeddings, temperature=1.0):
        scores = self.net(user_embeddings)
        scores = scores.squeeze(-1)
        scaled_scores = scores / temperature
        assignment_probs = F.softmax(scaled_scores, dim=1)

        return assignment_probs
    
class PowerAllocationHead(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64, total_power=1.0):
        super().__init__()
        self.total_power = total_power
        
        # Increase input dimension to include assignment probability
        input_dim = embed_dim + 1 
        
        self.power_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1), 
            nn.Sigmoid() # Raw "scores" in [0,1]
        )

    def forward(self, embeddings, assignment_probs):
        # 1. Prepare Input
        probs_expanded = assignment_probs.unsqueeze(-1)
        x = torch.cat([embeddings, probs_expanded], dim=-1)
        
        # 2. Predict Raw Scores
        raw_scores = self.power_net(x).squeeze(-1) + 1e-9
        
        # 3. FORCE FULL POWER USAGE (Softmax-style normalization)
        # Sum of scores per sample
        total_score = torch.sum(raw_scores, dim=(1, 2), keepdim=True)
        
        # Normalize: fractions always sum to 1.0
        power_fractions = raw_scores / total_score
        
        # Scale to Budget
        allocated_power = power_fractions * self.total_power
        
        return allocated_power
    
class WrapperAllocation(nn.Module):
    def __init__(self, tokenizer, encoder, assignment_head, allocation_head, emb_dim: int = 128):
        """
        Wrapper for the Carrier Selection Task

        """
        super().__init__()

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.assignment_head = assignment_head
        self.allocation_head = allocation_head
        self.emb_dim = emb_dim

        self.patch_rows = tokenizer.patch_rows
        self.patch_cols = tokenizer.patch_cols

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

    def forward(self, channels, temperature=1.0):
        """

        Inputs:
            channels: Shape (Samples, Users, Antennas, Subcarriers)
        """
        # 1. Extract channel shape
        B, K, M, SC = channels.shape

        # 2. Create tokens
        # Shape: [B, K, Seq_length, Features]
        tokens = self.tokenizer(channels)
        _, _, S, F = tokens.shape

        # 3. Flatten for Enconder
        # Shape: [B*K, Seq_length, Features]
        encoder_input = tokens.view(B*K, S, F)

        # 4. Encoder
        embeddings, _ = self.encoder(encoder_input)
        channel_emb = embeddings[:, 1:, :]

        # 5. Vertical and horizontal patches
        patch_width_flat = self.patch_cols * 2
        total_width_flat = SC * 2

        # Padding
        pad_rows = (self.patch_rows - (M % self.patch_rows)) % self.patch_rows
        num_v = (M + pad_rows) // self.patch_rows

        pad_cols = (patch_width_flat - (total_width_flat % patch_width_flat)) % patch_width_flat
        num_h = (total_width_flat + pad_cols) // patch_width_flat

        # 6. Recreate grid
        grid = channel_emb.view(B*K, num_v, num_h, -1)
        freq_features = torch.mean(grid, dim=1)

        # 7. Reshape for tasks
        user_embeddings = freq_features.view(B, K, num_h, -1)

        # 8. Task heads
        assignment_probs = self.assignment_head(user_embeddings, temperature=temperature)

        power_values = self.allocation_head(user_embeddings, assignment_probs)

        return assignment_probs, power_values