import torch.nn as nn
from torch.functional import F

class RegressionHead(nn.Module):
    def __init__(self, d_model, num_subcarriers, hidden_dim=512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), 
            nn.ReLU(),                
            nn.Dropout(0.2),          
            nn.Linear(hidden_dim, hidden_dim//2),     
            nn.ReLU(),                 
            nn.Dropout(0.2),          
            nn.Linear(hidden_dim//2, num_subcarriers)      
        )

    def forward(self, x):
        """
        Returns normalized power weights. The total power sums to 1 across all subcarriers/users.
        
        Inputs:
        x (torch.tenso): The embeddings sequence [B, User, Embedding dimension]

        Outputs:
        power_weights (torch.tensor): Normalized weights [Batch, Users, Subcarriers]
        """
        # 1. MLP Pass
        # PyTorch Linear layers automatically work on the last dimension, 
        # so [B, K, D] -> [B, K, S] works without flattening.
        raw_scores = self.net(x)

        B, K, S = raw_scores.shape

        # 2. Flatten K and S dimensions
        # We merge Users and Subcarriers into one pool: [B, K*S]
        flat_scores = raw_scores.view(B, -1)
        
        # 3. Apply Global Softmax
        weights = F.softmax(flat_scores, dim=1)

        # 4. Reshape back
        power_weights = weights.view(B, K, S)
        
        return power_weights
    
class Wrapper(nn.Module):
    def __init__(self,
                 model,
                 task_head):
        super().__init__()

        self.encoder = model
        self.task_head = task_head

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
        # Embeddings shape: [B*K, S, d_model]
        embeddings, _ = self.encoder(x)

        # 4. Extract CLS Token
        # Shape: [B*K, d_model]
        cls_embedding = embeddings[:, 0, :]

        # 5. Reshape to [Batch, Users, D] BEFORE the head
        # This is the key change. We reconstruct the user dimension here.
        cls_structured = cls_embedding.view(B, K, -1)

        # 6. Head Pass
        # The head now takes the structured data and returns normalized power
        power_weights = self.task_head(cls_structured)

        # 7. Permute: [B, K, S] -> [B, S, K]
        power_weights = power_weights.permute(0, 2, 1)

        return power_weights