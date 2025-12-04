import torch.nn as nn
from torch.functional import F

class RegressionHead(nn.Module):
    def __init__(self, d_model, n_carriers, n_users, n_antennas, hidden_dim=256):
        super().__init__()
        self.n_carriers = n_carriers
        self.n_users = n_users
        self.n_antennas = n_antennas
        
        self.net = nn.Sequential(
            nn.Linear(d_model, 512), 
            nn.ReLU(),                
            nn.Dropout(0.1),          
            nn.Linear(512, 256),     
            nn.ReLU(),                 
            nn.Dropout(0.1),          
            nn.Linear(256, n_users * n_antennas)      
        )
        # MLP Head
        #self.net = nn.Sequential(
        #    nn.Linear(d_model, hidden_dim),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim, n_users * n_antennas)
        #)

    def forward(self, x, p_max):
        # x: [Batch, S, d_model]
        batch_size, s_len, _ = x.shape
        
        raw_val = self.net(x)
        
        # [Batch, S * n_users * n_antennas]
        flat_val = raw_val.view(batch_size, -1)
        
        # Softmax activation function - Meets power constrains
        power_dist = F.softmax(flat_val, dim=1)
        
        # Scale by total power
        final_flat = power_dist * p_max
        
        # Reshape to [Batch, S, n_users, n_antennas]
        final_power = final_flat.view(batch_size, self.n_carriers, self.n_users, self.n_antennas)
        
        return final_power