def sum_rate_loss_per_subcarrier(H, V_directions, power_map, sigma2=1.0):
    B, N, K, M = H.shape
    
    # 1. Scale Precoder
    p_reshaped = power_map.unsqueeze(2) # (B, N, 1, K)
    V = V_directions * torch.sqrt(p_reshaped)
    
    # 2. Effective Channel
    HV = torch.matmul(H, V)
    
    # 3. Power Calculations
    # Extract diagonals (Signal Power)
    sig_power = torch.diagonal(HV, dim1=-2, dim2=-1).abs()**2 
    
    # Total Power Received
    total_rx = torch.sum(HV.abs()**2, dim=-1)
    
    # --- CRITICAL FIX HERE ---
    # Interference = Total - Signal
    # In Float32, this subtraction can result in -0.0000001 due to precision errors.
    interference = total_rx - sig_power
    
    # Clamp interference to be at least 0.0
    interference = torch.clamp(interference, min=1e-9)
    
    # Denominator
    noise_plus_inter = interference + sigma2
    
    # 4. Rate Calculation
    # Add epsilon to SINR to prevent log(0) if signal is 0
    sinr = sig_power / noise_plus_inter
    rates = torch.log2(1 + sinr)
    
    # Sum over Subcarriers (N) and Users (K)
    total_sum_rate = torch.sum(rates, dim=(1, 2))
    
    return -torch.mean(total_sum_rate)


import torch

def get_zf_precoder_ofdm(H, diagonal_loading=1e-4, normalize=False):
    """
    Generates ZF Precoder matrices for an OFDM system (4D Tensor).
    
    Args:
        H: (Batch, Subcarriers, Users, Antennas) Complex Tensor
           Dimensions: (B, N, K, M)
        diagonal_loading: float - Stability factor (RZF).
        normalize: bool - If True, output columns (directions) have unit norm.
        
    Returns:
        V: (Batch, Subcarriers, Antennas, Users) Complex Tensor
           Dimensions: (B, N, M, K)
    """
    B, N, K, M = H.shape
    
    # Constraint Check: Antennas >= Users
    assert M >= K, f"ZF requires M ({M}) >= K ({K})"

    # --- 1. Conjugate Transpose (Hermitian) ---
    # We swap the last two dimensions (User <-> Antenna)
    # H: (B, N, K, M) -> H_H: (B, N, M, K)
    H_H = H.conj().transpose(-2, -1)
    
    # --- 2. Gram Matrix Calculation ---
    # Matrix Multiplication applies to the last two dimensions
    # (..., K, M) @ (..., M, K) -> (..., K, K)
    # Result shape: (B, N, K, K)
    gram_matrix = torch.matmul(H, H_H)
    
    # --- 3. Regularization (Diagonal Loading) ---
    # Create Identity (K, K) and move to correct device
    eye = torch.eye(K, device=H.device, dtype=H.dtype)
    regularized_gram = gram_matrix + diagonal_loading * eye
    
    # Safe Solve
    try:
        inv_gram = torch.linalg.solve(regularized_gram, eye.expand(B, N, K, K))
    except RuntimeError:
        # Fallback if solve fails: Use pseudo-inverse (slower but robust)
        # This handles singular matrices better
        inv_gram = torch.linalg.pinv(regularized_gram)

    V_raw = torch.matmul(H_H, inv_gram)

    if normalize:
        # Add larger epsilon to avoid div by zero
        column_norms = torch.norm(V_raw, dim=-2, keepdim=True)
        V_out = V_raw / (column_norms + 1e-8)
    else:
        V_out = V_raw

    return V_out

EPS = 1e-12

class PowerAllocator(nn.Module):
    """A regressor that processes channel embeddings for power allocation only.

    This model assumes that the beamforming directions are fixed (e.g., using
    the optimal WMMSE directions) and only learns to predict the power scalars.
    Input shape: (B, K, P, D) -> (Batch, Users, Patches, Embedding Dimension)
    """
    def __init__(self,
                 emb_dim: int,
                 n_patches: int,
                 n_users: int,
                 n_subcarriers: int,
                 n_antennas: int, # Added for consistency, though not used in heads
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dropout_rate: float = 0.1):
        
        super().__init__()
        self.n_patches = n_patches
        self.n_users = n_users
        self.n_subcarriers = n_subcarriers
        self.n_antennas = n_antennas
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        # --- Model Layers (identical to your original model) ---
        self.emb_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.encoder_ln = nn.LayerNorm(hidden_dim)
        self.patch_pos_embedding = nn.Parameter(torch.randn(1, n_patches, 1, hidden_dim))
        self.patch_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.patch_ln = nn.LayerNorm(hidden_dim)
        self.user_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.user_ln = nn.LayerNorm(hidden_dim)
        self.patch_to_sc_mapper = nn.Sequential(
            nn.Linear(n_patches, n_subcarriers),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.patch_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.patch_ffn_ln = nn.LayerNorm(hidden_dim)
        self.user_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.user_ffn_ln = nn.LayerNorm(hidden_dim)

        # --- Output Head (Power Only) ---
        self.pow_head = nn.Linear(hidden_dim, 1) # Outputs a single scalar for power allocation

    def forward(self, inputs: torch.Tensor):
        # B: batch, K: users, P: patches, D: embedding dim
        B, _, _, _ = inputs.shape
        P, K, SC = self.n_patches, self.n_users, self.n_subcarriers

        # --- 1. Input Encoding & 2. Attention (identical to your original model) ---
        x = inputs.permute(0, 2, 1, 3)
        x = x.reshape(B * P * K, self.emb_dim)
        features = self.emb_encoder(x)
        features = self.encoder_ln(features)
        features = features.view(B, P, K, self.hidden_dim)
        features = features + self.patch_pos_embedding
        z = features.permute(0, 2, 1, 3).reshape(B * K, P, self.hidden_dim)
        attn_out, _ = self.patch_attn(z, z, z)
        z = self.patch_ln(z + attn_out)
        ffn_out = self.patch_ffn(z)
        z = self.patch_ffn_ln(z + ffn_out)
        z = z.view(B, K, P, self.hidden_dim).permute(0, 2, 1, 3).reshape(B * P, K, self.hidden_dim)
        attn_out, _ = self.user_attn(z, z, z)
        z = self.user_ln(z + attn_out)
        ffn_out = self.user_ffn(z)
        z = self.user_ffn_ln(z + ffn_out)

        # --- 3. Map from Patches to Subcarriers (identical) ---
        z = z.view(B, P, K, self.hidden_dim).permute(0, 2, 3, 1)
        z = self.patch_to_sc_mapper(z)
        z = z.permute(0, 3, 1, 2)

        # --- 4. Output Head (Power Only) ---
        z = z.reshape(B * SC * K, self.hidden_dim)
        pow_raw = self.pow_head(z)

        # --- 5. Reshape and Normalize Power ---
        pow_raw = pow_raw.view(B, -1) # Shape: (B, SC * K)
        # Apply softmax over all users and subcarriers for each batch item
        alpha_flat = F.softmax(pow_raw, dim=-1)
        alpha = alpha_flat.view(B, SC, K) # Reshape back to (B, SC, K)

        return alpha


class PowerAllocator(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 n_patches: int, 
                 n_users: int, 
                 n_subcarriers: int, 
                 hidden_dim: int = 256, 
                 num_heads: int = 4, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.n_patches = n_patches
        self.n_users = n_users
        self.n_subcarriers = n_subcarriers
        
        # --- 1. Embedding Encoder ---
        self.emb_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # --- 2. Attention Blocks ---
        # FIXED: Shape is now (1, 1, P, H) to match (B, K, P, H)
        self.patch_pos_emb = nn.Parameter(torch.randn(1, 1, n_patches, hidden_dim))
        
        self.patch_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.user_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # --- 3. Mapping Patches -> Subcarriers ---
        self.patch_to_sc = nn.Linear(n_patches, n_subcarriers)
        
        # --- 4. Cost Injection Fusion ---
        # Input to head: Hidden_Dim + 1 (The ZF cost scalar)
        self.pow_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, embeddings, zf_costs):
        """
        embeddings: (B, K, P, D)
        zf_costs: (B, K, SC)
        """
        B = embeddings.shape[0]
        P, K, SC = self.n_patches, self.n_users, self.n_subcarriers

        # 1. Encode 
        # (B, K, P, D) -> (B, K, P, H)
        x = self.emb_encoder(embeddings) 
        
        # 2. Add Positional Info (Broadcasts correctly now)
        # (B, K, P, H) + (1, 1, P, H) -> Works!
        x = x + self.patch_pos_emb 

        # 3. Patch Attention (Per User)
        # Merge Batch and Users: (B*K, P, H)
        x_flat = x.view(B*K, P, -1)
        attn_out, _ = self.patch_attn(x_flat, x_flat, x_flat)
        x_flat = self.ln1(x_flat + attn_out)

        # 4. User Attention (Per Patch)
        # Reshape to (B, K, P, H) -> Permute to (B, P, K, H) -> Flatten (B*P, K, H)
        x_user = x_flat.view(B, K, P, -1).permute(0, 2, 1, 3).reshape(B*P, K, -1)
        attn_out, _ = self.user_attn(x_user, x_user, x_user)
        x_user = self.ln2(x_user + attn_out)

        # 5. Map to Subcarriers
        # Current shape: (B*P, K, H) -> View (B, P, K, H) -> Permute (B, K, H, P)
        x = x_user.view(B, P, K, -1).permute(0, 2, 3, 1)
        
        # Linear project P -> SC: (B, K, H, SC)
        x = self.patch_to_sc(x)
        
        # Permute to (B, K, SC, H) to align with zf_costs
        x = x.permute(0, 1, 3, 2)

        # 6. Fuse with ZF Costs
        # zf_costs is (B, K, SC). Unsqueeze to (B, K, SC, 1)
        costs = zf_costs.unsqueeze(-1)
        # Concatenate: (B, K, SC, H+1)
        features = torch.cat([x, costs], dim=-1)

        # 7. Power Head
        # (B, K, SC, 1)
        logits = self.pow_head(features).squeeze(-1)

        # 8. Global Softmax
        # Flatten to (B, K*SC)
        flat_logits = logits.view(B, -1)
        probs = F.softmax(flat_logits, dim=-1)
        
        # Reshape back to (B, SC, K) for the loss function
        # Logic: (B, K*SC) -> (B, K, SC) -> Transpose -> (B, SC, K)
        power_map = probs.view(B, K, SC).transpose(1, 2)
        
        return power_map
    


def compute_exact_sum_rate(H, V, sigma2=1.0):
    """
    Calculates Sum Rate given Channel H and Precoder V.
    
    Args:
        H: (B, N, K, M) - Channel
        V: (B, N, M, K) - Precoder (WMMSE or DL-ZF)
        sigma2: Noise variance
    
    Returns:
        sum_rates: (B,) - Sum Rate in bps/Hz per sample
    """
    # 1. Effective Channel (Signal + Interference)
    # (B, N, K, M) @ (B, N, M, K) -> (B, N, K, K)
    HV = torch.matmul(H, V)
    
    # 2. Signal Power (Diagonal elements magnitude squared)
    # dim -2 is row (User Rx), dim -1 is col (Stream Tx)
    sig_power = torch.diagonal(HV, dim1=-2, dim2=-1).abs()**2 # (B, N, K)
    
    # 3. Interference + Noise
    # Sum of squared magnitude of all incoming signals at User K
    total_rx_power = torch.sum(HV.abs()**2, dim=-1) # (B, N, K)
    
    # Interference = Total - Signal
    interference = total_rx_power - sig_power
    
    # 4. Capacity Formula
    # Rate = log2(1 + Signal / (Interference + Noise))
    sinr = sig_power / (interference + sigma2)
    rates = torch.log2(1 + sinr)
    
    # 5. Sum over Subcarriers (N) and Users (K)
    total_rate = torch.sum(rates, dim=(1, 2))
    
    return total_rate

SEED = 42

# --- Training Configuration ---
N_EPOCHS = 15
LEARNING_RATE = 1e-4
HIDDEN = 256
NUM_HEADS = 4



DATASETS = [#"sim_city_6_miami_fürstenwalde",
            #"sim_city_6_miami_säckingen",
            #"sim_city_6_miami_deggendorf",
            "sim_city_6_miami_warendorf"]


for dataset in DATASETS:

    parameters = get_parameters(f"../data/{dataset}/parameters.txt")

    N_SAMPLES = parameters["samples"]
    
    N_USERS = parameters["users"]
    P_TOTAL = parameters["p_total"]
    NOISE_VARIANCE = parameters["sigma2"]
    INPUT_TYPES = parameters["input_types"]

    channels_dataset = np.load(f"../data/{dataset}/channels.npy")
    precoder_dataset = np.load(f"../data/../data/{dataset}/precoder.npy")

    channels_dataset = torch.tensor(channels_dataset, dtype=torch.complex64)
    precoder_dataset = torch.tensor(precoder_dataset, dtype=torch.complex64)
    # Extract dimensions from data
    _, SC, K, M = channels_dataset.shape
    print("Number of subcarriers:", SC, "Number of Users:", K, "Number of TX antennas:", M)
    
    for input_type in INPUT_TYPES:
        embeddings_dataset = np.load(f"../data/{dataset}/{input_type}.npy")
        embeddings_dataset = torch.tensor(embeddings_dataset, dtype=torch.float32)

        # Unpack dimensions from embeddings tensor: (B, K, S, D)
        _, _, S, EMBED_DIM = embeddings_dataset.shape
        # The shape is now (N_SAMPLES, N_USERS, N_PATCHES, EMBED_DIM), so we unpack it
        print("\tInput Type:", input_type, "Number of patches:", S, "Embedding dimension:", EMBED_DIM)
        
        for fraction in TRAINING_FRACTIONS:
            split = [0.7*fraction, 0.2, 0.1, 0.7*(1.0-fraction)]
            # Prepare data loaders based on the input type and split ratio
            train_loader, val_loader, test_loader = prepare_loaders(channels_dataset, embeddings_dataset, precoder_dataset, split, seed=SEED)

            power_model = PowerAllocator(
                             n_patches=S,
                             n_users=K,
                             n_subcarriers=SC,
                             emb_dim=EMBED_DIM,
                             hidden_dim=HIDDEN,
                             num_heads=NUM_HEADS
                             ).to(INFERENCE_DEVICE)

            optimizer = torch.optim.Adam(power_model.parameters(), lr=LEARNING_RATE)

            for epoch in range(N_EPOCHS):
                power_model.train()
                train_loss = 0.0
                for channel, embeddings, precoder in train_loader:
                    channel, embeddings, precoder = channel.to(INFERENCE_DEVICE), embeddings.to(INFERENCE_DEVICE), precoder.to(INFERENCE_DEVICE)

                    optimizer.zero_grad()

                    zf_precoder = get_zf_precoder_ofdm(channel, normalize=False)
                    zf_norms = torch.norm(zf_precoder, dim=-2).squeeze(-2)
                    zf_costs = zf_norms.transpose(1, 2)
                    zf_dirs = zf_precoder / (zf_norms.unsqueeze(2) + 1e-10)

                    powers = power_model(embeddings, zf_costs) * P_TOTAL
                    
                    loss = sum_rate_loss_per_subcarrier(channel, zf_dirs, powers, NOISE_VARIANCE)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)

                power_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for channel, embeddings, precoder in val_loader:
                        channel, embeddings, precoder = channel.to(INFERENCE_DEVICE), embeddings.to(INFERENCE_DEVICE), precoder.to(INFERENCE_DEVICE)

                        zf_precoder = get_zf_precoder_ofdm(channel, normalize=False)
                        zf_norms = torch.norm(zf_precoder, dim=-2).squeeze(-2)
                        zf_costs = zf_norms.transpose(1, 2)
                        zf_dirs = zf_precoder / (zf_norms.unsqueeze(2) + 1e-10)

                        powers = power_model(embeddings, zf_costs) * P_TOTAL

                        loss = sum_rate_loss_per_subcarrier(channel, zf_dirs, powers, NOISE_VARIANCE)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

            power_model.eval()
            test_loss = 0.0
            wmmse_rate = 0.0
            total_samples = 0.0
            with torch.no_grad():
                for channel, embeddings, precoder in test_loader:
                    channel, embeddings, precoder = channel.to(INFERENCE_DEVICE), embeddings.to(INFERENCE_DEVICE), precoder.to(INFERENCE_DEVICE)
                    total_samples += channel.size(0)

                    zf_precoder = get_zf_precoder_ofdm(channel, normalize=False)
                    zf_norms = torch.norm(zf_precoder, dim=-2).squeeze(-2)
                    zf_costs = zf_norms.transpose(1, 2)
                    zf_dirs = zf_precoder / (zf_norms.unsqueeze(2) + 1e-10)

                    powers = power_model(embeddings, zf_costs) * P_TOTAL

                    loss = sum_rate_loss_per_subcarrier(channel, zf_dirs, powers, NOISE_VARIANCE)
                    test_loss += loss.item() * channel.size(0)

                    wmmse_rate += compute_exact_sum_rate(channel, precoder, NOISE_VARIANCE).sum().item()

            avg_wmmse_rates = wmmse_rate /total_samples
            avg_test_loss = -1 * test_loss / total_samples

            print(f"\t\tTraining Fraction: {fraction} Test rate: {avg_test_loss:.2f} Benchmark Rate: {avg_wmmse_rates:.2f} Performance Ratio: {avg_test_loss/avg_wmmse_rates:.2f}")

    print("\n")



import numpy as np
import torch
from scipy.optimize import minimize

def numerical_power_opt_batch(u_directions, H_channel, noise_var, P_total=1.0, maxiter=200):
    """
    Numerically optimize per-user power allocation for each batch sample,
    given fixed precoding directions, using sum-rate maximization.

    Parameters
    ----------
    u_directions : torch.Tensor
        Fixed beamforming directions, shape (B, S, K, M)
    H_channel : torch.Tensor
        Channel matrices, shape (B, S, K, M)
    noise_var : float
        Noise variance
    P_total : float
        Total transmit power per batch item (sum over all users & subcarriers)
    maxiter : int
        Maximum iterations for optimizer

    Returns
    -------
    sumrates : np.ndarray
        Optimized sum-rate per batch item, shape (B,)
    p_opt_list : list of np.ndarray
        Optimized power allocations per batch item, each shape (S, K)
    """
    B, S, K, M = u_directions.shape
    sumrates = np.zeros(B)
    p_opt_list = []

    for b in range(B):
        u_sample = u_directions[b:b+1].clone()  # (1, S, K, M)
        H_sample = H_channel[b:b+1].clone()     # (1, S, K, M)

        def objective(p_flat):
            p_tensor = torch.tensor(p_flat.reshape(1, S, K), dtype=torch.float32)
            with torch.no_grad():
                rate, _ = compute_sumrate_from_directions(u_sample, H_sample, p_tensor, noise_var)
            return -rate.item()

        # Initial guess: equal power
        p0 = np.ones(S*K) * (P_total / (S*K))

        # Bounds for each power scalar
        bounds = [(0, P_total) for _ in range(S*K)]

        # Equality constraint: total power = P_total
        constraints = {'type': 'eq', 'fun': lambda p: p.sum() - P_total}

        res = minimize(
            objective,
            p0,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-6, 'maxiter': maxiter, 'disp': False}
        )

        p_opt = res.x.reshape(S, K)
        sumrate_opt = -res.fun

        sumrates[b] = sumrate_opt
        p_opt_list.append(p_opt)

    return sumrates, p_opt_list

sum_rates = []
for batch in tqdm(test_loader, desc="Testing PowerAllocator"):
    channels_batch_gpu = batch[0]
    input_data_gpu = batch[1]
    u_wmmse_gpu = batch[2]
    p_target_gpu = batch[3]
    # Assume you have u_wmmse_gpu (B,S,K,M) and channels_batch_gpu (B,S,K,M)
    sumrates_opt, p_opt_batch = numerical_power_opt_batch(
        u_wmmse_gpu, channels_batch_gpu, noise_var=NOISE_VARIANCE, P_total=1.0
    )
    sum_rates.extend(sumrates_opt)
print("Average numerical optimum sum-rate:", np.mean(sum_rates))


import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepBeamformer(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 n_patches, 
                 n_users, 
                 n_subcarriers, 
                 n_antennas, # M is crucial here
                 hidden_dim=256, 
                 num_heads=4, 
                 dropout_rate=0.1):
        super().__init__()
        
        self.n_users = n_users
        self.n_subcarriers = n_subcarriers
        self.n_antennas = n_antennas

        # --- 1. Feature Extraction (Same as before) ---
        self.emb_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer Backbone
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=3
        )

        # --- 2. Subcarrier Mapping ---
        # Maps Patch dimension -> Subcarrier dimension
        self.patch_to_sc = nn.Linear(n_patches, n_subcarriers)

        # --- 3. The Beamforming Head ---
        # Instead of outputting 1 power scalar, we output (2 * M) values per User per Subcarrier.
        # 2 * M corresponds to Real and Imaginary parts for M antennas.
        self.beam_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_antennas * 2) # Output: (Real_1, Imag_1, Real_2, Imag_2...)
        )

    def forward(self, embeddings, P_max):
        """
        Args:
            embeddings: (B, K, P, D)
            P_max: float - Total power budget
        Returns:
            V: (B, N, M, K) Complex Tensor
        """
        B, K, P, D = embeddings.shape
        N = self.n_subcarriers
        M = self.n_antennas

        # A. Encode
        x = self.emb_encoder(embeddings.view(B * K, P, D)) # (BK, P, H)
        
        # B. Transformer (Process interactions)
        x = self.transformer(x) # (BK, P, H)

        # C. Map Patches to Subcarriers
        # (BK, P, H) -> (BK, H, P) -> Linear -> (BK, H, N) -> (BK, N, H)
        x = x.permute(0, 2, 1)
        x = self.patch_to_sc(x)
        x = x.permute(0, 2, 1) 

        # D. Predict Beamformers
        # Input: (BK, N, H) -> Output: (BK, N, M * 2)
        raw_beam = self.beam_head(x)
        
        # E. Reshape to (B, N, K, M, 2)
        # We separate Users (K) and Antennas (M) and Complex Parts (2)
        raw_beam = raw_beam.view(B, K, N, M, 2)
        
        # Permute to standard format: (B, N, M, K, 2)
        # (Users are columns of V)
        raw_beam = raw_beam.permute(0, 2, 3, 1, 4)

        # F. Construct Complex Tensor
        V_real = raw_beam[..., 0]
        V_imag = raw_beam[..., 1]
        V = torch.complex(V_real, V_imag) # (B, N, M, K)

        # --- G. Power Normalization (CRITICAL) ---
        # We must satisfy Sum(|V|^2) <= P_max
        
        # Calculate current total power of the raw predictions
        current_power = torch.sum(torch.abs(V)**2, dim=(1, 2, 3)) # Sum over N, M, K -> (B,)
        
        # Scaling factor
        # We use P_max + epsilon to avoid division by zero
        scale = torch.sqrt(P_max / (current_power + 1e-12))
        
        # Broadcast scale: (B,) -> (B, 1, 1, 1)
        V_final = V * scale.view(B, 1, 1, 1)

        return V_final
    
def beamforming_rate_loss(H, V_pred, sigma2=1.0):
    """
    Calculates Negative Sum Rate for Direct Beamforming.
    H: (B, N, K, M)
    V_pred: (B, N, M, K)
    """
    # 1. Effective Channel (Signal + Interference)
    # (B, N, K, M) @ (B, N, M, K) -> (B, N, K, K)
    HV = torch.matmul(H, V_pred)
    
    # 2. Signal Power (Diagonal elements magnitude squared)
    sig_power = torch.diagonal(HV, dim1=-2, dim2=-1).abs()**2 # (B, N, K)
    
    # 3. Total Power Received
    total_rx = torch.sum(HV.abs()**2, dim=-1) # (B, N, K)
    
    # 4. Interference + Noise
    # Clamp to avoid negative values due to float32 precision errors
    interference = torch.clamp(total_rx - sig_power, min=1e-9)
    noise_plus_inter = interference + sigma2
    
    # 5. Rate Calculation
    sinr = sig_power / noise_plus_inter
    rates = torch.log2(1 + sinr)
    
    # 6. Sum over Subcarriers and Users
    sum_rate = torch.sum(rates, dim=(1, 2))
    
    # Return Negative Mean for Minimization
    return -torch.mean(sum_rate)

import torch
import torch.optim as optim
import numpy as np

# --- MAIN LOOP ---

for dataset in DATASETS:
    # ... Load Parameters ...
    parameters = get_parameters(f"../data/{dataset}/parameters.txt")
    N_SAMPLES = parameters["samples"]
    N_USERS = parameters["users"]
    P_TOTAL = parameters["p_total"]
    NOISE_VARIANCE = parameters["sigma2"]
    INPUT_TYPES = parameters["input_types"]

    # Load Data
    channels_dataset = np.load(f"../data/{dataset}/channels.npy")
    precoder_dataset = np.load(f"../data/../data/{dataset}/precoder.npy")

    channels_dataset = torch.tensor(channels_dataset, dtype=torch.complex64)
    precoder_dataset = torch.tensor(precoder_dataset, dtype=torch.complex64)
    
    # --- CRITICAL: CHANNEL NORMALIZATION ---
    # Deep Beamforming fails if channel values are tiny (e.g., 1e-6). 
    # We normalize average channel energy to 1.0.
    # This stabilizes gradients.
    scale_factor = torch.sqrt(torch.mean(torch.abs(channels_dataset)**2))
    channels_dataset = channels_dataset / scale_factor
    # Note: Since we scale H, the effective Noise Variance implies the SNR.
    # If Noise=1.0 and H is normalized, SNR is roughly P_TOTAL.
    
    _, SC, K, M = channels_dataset.shape
    print(f"Dataset: {dataset} | Norm Factor: {scale_factor:.2e}")

    for input_type in INPUT_TYPES:
        embeddings_dataset = np.load(f"../data/{dataset}/{input_type}.npy")
        embeddings_dataset = torch.tensor(embeddings_dataset, dtype=torch.float32)
        _, _, S, EMBED_DIM = embeddings_dataset.shape

        for fraction in TRAINING_FRACTIONS:
            split = [0.7*fraction, 0.7*(1.0-fraction), 0.2, 0.1]
            
            # No ZF inputs needed here
            train_loader, val_loader, test_loader = prepare_loaders(
                channels_dataset, embeddings_dataset, precoder_dataset, split, seed=SEED
            )

            # Initialize DeepBeamformer
            model = DeepBeamformer(
                emb_dim=EMBED_DIM,
                n_patches=S,
                n_users=K,
                n_subcarriers=SC,
                n_antennas=M,       # Needed for output shape
                hidden_dim=HIDDEN,
                num_heads=NUM_HEADS
            ).to(INFERENCE_DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            # Scheduler helps beamforming convergence significantly
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            for epoch in range(N_EPOCHS):
                # ==========================
                # TRAINING
                # ==========================
                model.train()
                train_loss = 0.0
                
                for channel, embeddings, _ in train_loader: # Unpack only what we need
                    channel = channel.to(INFERENCE_DEVICE)
                    embeddings = embeddings.to(INFERENCE_DEVICE)

                    optimizer.zero_grad()

                    # 1. Forward Pass: Model predicts V directly
                    # V_pred shape: (Batch, SC, M, K)
                    # Internal logic of model handles P_TOTAL normalization
                    V_pred = model(embeddings, P_TOTAL)

                    # 2. Loss Calculation
                    loss = beamforming_rate_loss(channel, V_pred, NOISE_VARIANCE)
                    
                    # 3. Backward & Safe Step
                    loss.backward()
                    
                    # Clip gradients to prevent exploding gradients (common in complex nets)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # ==========================
                # VALIDATION
                # ==========================
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for channel, embeddings, _ in val_loader:
                        channel = channel.to(INFERENCE_DEVICE)
                        embeddings = embeddings.to(INFERENCE_DEVICE)

                        V_pred = model(embeddings, P_TOTAL)
                        loss = beamforming_rate_loss(channel, V_pred, NOISE_VARIANCE)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                
                # Step Scheduler
                scheduler.step(avg_val_loss)

            # ==========================
            # TESTING
            # ==========================
            model.eval()
            test_loss = 0.0
            wmmse_rate = 0.0
            total_samples = 0
            
            with torch.no_grad():
                for channel, embeddings, wmmse_precoder in test_loader:
                    channel = channel.to(INFERENCE_DEVICE)
                    embeddings = embeddings.to(INFERENCE_DEVICE)
                    wmmse_precoder = wmmse_precoder.to(INFERENCE_DEVICE)

                    total_samples += channel.size(0)

                    # 1. Model Rate
                    V_pred = model(embeddings, P_TOTAL)
                    loss = beamforming_rate_loss(channel, V_pred, NOISE_VARIANCE)
                    test_loss += loss.item() * channel.size(0)

                    # 2. WMMSE Benchmark Rate
                    # Note: We use the same scaled channel for fair comparison!
                    # compute_exact_sum_rate should use the same math logic as loss
                    batch_wmmse_rates = compute_exact_sum_rate(channel, wmmse_precoder, NOISE_VARIANCE)
                    wmmse_rate += batch_wmmse_rates.sum().item()

            avg_wmmse_rate = wmmse_rate / total_samples
            avg_dl_rate = -1 * test_loss / total_samples

            print(f"Input type: {input_type} | Fraction: {fraction} | DL Rate: {avg_dl_rate:.2f} | WMMSE Rate: {avg_wmmse_rate:.2f} | Ratio: {avg_dl_rate/avg_wmmse_rate:.2f}")

    print("\n")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# --- CONFIGURATION ---
INFERENCE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ==========================================
# 1. HELPER CLASSES & FUNCTIONS
# ==========================================

class PhaseInvariantLoss(nn.Module):
    """
    Calculates 1 - |CosineSimilarity|.
    This ignores global phase rotation, which is crucial for beamforming.
    """
    def __init__(self):
        super().__init__()

    def forward(self, v_pred, v_target):
        # Inner product over M antennas (dim -2)
        # v_pred: (B, N, M, K)
        inner_prod = torch.sum(v_pred * v_target.conj(), dim=-2)
        
        # Magnitudes
        norm_pred = torch.norm(v_pred, dim=-2)
        norm_target = torch.norm(v_target, dim=-2)
        
        # Cosine Similarity (Abs ignores phase)
        cosine_sim = torch.abs(inner_prod) / (norm_pred * norm_target + 1e-10)
        
        # Loss = 1 - Mean Similarity
        return 1.0 - torch.mean(cosine_sim)

class DeepBeamformer(nn.Module):
    def __init__(self, emb_dim, n_patches, n_users, n_subcarriers, n_antennas, hidden_dim=256, num_heads=4):
        super().__init__()
        self.n_subcarriers = n_subcarriers
        self.n_antennas = n_antennas

        self.emb_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer for interactions
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=3
        )

        # Map Patches -> Subcarriers
        self.patch_to_sc = nn.Linear(n_patches, n_subcarriers)

        # Head: Predicts Complex V (Real, Imag) per Antenna
        self.beam_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_antennas * 2) 
        )

    def forward(self, embeddings, P_max):
        B, K, P, D = embeddings.shape
        N, M = self.n_subcarriers, self.n_antennas

        # 1. Encode
        x = self.emb_encoder(embeddings.view(B * K, P, D)) # (BK, P, H)
        x = self.transformer(x)
        
        # 2. Map Patches -> Subcarriers
        x = x.permute(0, 2, 1) # (BK, H, P)
        x = self.patch_to_sc(x)
        x = x.permute(0, 2, 1) # (BK, N, H)

        # 3. Predict Beamformers
        raw_beam = self.beam_head(x)
        
        # 4. Reshape to (B, N, M, K) Complex
        raw_beam = raw_beam.view(B, K, N, M, 2).permute(0, 2, 3, 1, 4)
        V = torch.complex(raw_beam[..., 0], raw_beam[..., 1])

        # 5. Power Normalization (Global Sum Power Constraint)
        current_power = torch.sum(torch.abs(V)**2, dim=(1, 2, 3)) # (B,)
        scale = torch.sqrt(P_max / (current_power + 1e-12))
        V_final = V * scale.view(B, 1, 1, 1)

        return V_final

def compute_exact_sum_rate(H, V, sigma2=1.0):
    # (B, N, K, M) @ (B, N, M, K) -> (B, N, K, K)
    HV = torch.matmul(H, V)
    sig_power = torch.diagonal(HV, dim1=-2, dim2=-1).abs()**2
    total_rx = torch.sum(HV.abs()**2, dim=-1)
    interference = torch.clamp(total_rx - sig_power, min=1e-9)
    rates = torch.log2(1 + sig_power / (interference + sigma2))
    return torch.sum(rates, dim=(1, 2))

def prepare_fixed_test_loaders(channels, embeddings, wmmse_precoders, fraction, seed=42):
    dataset = torch.utils.data.TensorDataset(channels, embeddings, wmmse_precoders)
    total_len = len(dataset)
    
    # 1. Isolate Test Set (Fixed 10%)
    test_len = int(0.1 * total_len)
    remaining_len = total_len - test_len
    
    gen = torch.Generator().manual_seed(seed)
    remaining_ds, test_ds = torch.utils.data.random_split(dataset, [remaining_len, test_len], generator=gen)
    
    # 2. Split Remaining into Train/Val based on Fraction
    val_len = int(0.2 * remaining_len)
    max_train_len = remaining_len - val_len
    current_train_len = int(max_train_len * fraction)
    unused_len = max_train_len - current_train_len
    
    train_ds, val_ds, unused_ds = torch.utils.data.random_split(
        remaining_ds, [current_train_len, val_len, unused_len], generator=gen
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ==========================================
# 2. MAIN EXECUTION LOOP
# ==========================================

# Define Loss Function
criterion = PhaseInvariantLoss().to(INFERENCE_DEVICE)

for dataset in DATASETS:
    parameters = get_parameters(f"../data/{dataset}/parameters.txt")
    N_USERS = parameters["users"]
    P_TOTAL = parameters["p_total"]
    NOISE_VARIANCE = parameters["sigma2"]
    INPUT_TYPES = parameters["input_types"]

    # Load Data
    channels_dataset = np.load(f"../data/{dataset}/channels.npy")
    precoder_dataset = np.load(f"../data/../data/{dataset}/precoder.npy") # WMMSE Targets

    channels_dataset = torch.tensor(channels_dataset, dtype=torch.complex64)
    precoder_dataset = torch.tensor(precoder_dataset, dtype=torch.complex64)
    
    # --- CRITICAL: NORMALIZE CHANNELS ---
    # Deep Beamforming fails on tiny numbers. Normalize channel energy to 1.
    scale_factor = torch.sqrt(torch.mean(torch.abs(channels_dataset)**2))
    channels_dataset = channels_dataset / scale_factor
    print(f"Dataset: {dataset} | Channel Scale Factor: {scale_factor:.2e}")

    _, SC, K, M = channels_dataset.shape
    
    for input_type in INPUT_TYPES:
        embeddings_dataset = np.load(f"../data/{dataset}/{input_type}.npy")
        embeddings_dataset = torch.tensor(embeddings_dataset, dtype=torch.float32)
        _, _, S, EMBED_DIM = embeddings_dataset.shape
        
        print(f"\tInput: {input_type} | Emb Dim: {EMBED_DIM}")

        for fraction in TRAINING_FRACTIONS:
            # Use FIXED test set loader
            train_loader, val_loader, test_loader = prepare_fixed_test_loaders(
                channels_dataset, embeddings_dataset, precoder_dataset, fraction, seed=SEED
            )

            model = DeepBeamformer(
                emb_dim=EMBED_DIM, n_patches=S, n_users=K, 
                n_subcarriers=SC, n_antennas=M, hidden_dim=HIDDEN, num_heads=NUM_HEADS
            ).to(INFERENCE_DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

            for epoch in range(N_EPOCHS):
                # --- TRAIN (Supervised) ---
                model.train()
                train_loss = 0.0
                
                for channel, embeddings, wmmse_target in train_loader:
                    embeddings = embeddings.to(INFERENCE_DEVICE)
                    wmmse_target = wmmse_target.to(INFERENCE_DEVICE)

                    optimizer.zero_grad()
                    
                    # Predict V
                    v_pred = model(embeddings, P_TOTAL)
                    
                    # Loss: Distance to WMMSE Target
                    loss = criterion(v_pred, wmmse_target)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # --- VALIDATION ---
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for channel, embeddings, wmmse_target in val_loader:
                        channel = channel.to(INFERENCE_DEVICE)
                        embeddings = embeddings.to(INFERENCE_DEVICE)
                        wmmse_target = wmmse_target.to(INFERENCE_DEVICE)

                        v_pred = model(embeddings, P_TOTAL)
                        loss = criterion(v_pred, wmmse_target)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

            # --- TEST ---
            model.eval()
            total_samples = 0
            dl_sum_rate = 0.0
            wmmse_sum_rate = 0.0
            
            with torch.no_grad():
                for channel, embeddings, wmmse_target in test_loader:
                    channel = channel.to(INFERENCE_DEVICE)
                    embeddings = embeddings.to(INFERENCE_DEVICE)
                    wmmse_target = wmmse_target.to(INFERENCE_DEVICE)
                    
                    total_samples += channel.size(0)

                    # Predict
                    v_pred = model(embeddings, P_TOTAL)

                    # Calculate Rates (Using normalized channel & P_total)
                    dl_rates = compute_exact_sum_rate(channel, v_pred, NOISE_VARIANCE)
                    wmmse_rates = compute_exact_sum_rate(channel, wmmse_target, NOISE_VARIANCE)
                    
                    dl_sum_rate += dl_rates.sum().item()
                    wmmse_sum_rate += wmmse_rates.sum().item()

            avg_dl_rate = dl_sum_rate / total_samples
            avg_wmmse_rate = wmmse_sum_rate / total_samples
            ratio = avg_dl_rate / avg_wmmse_rate

            print(f"\t\tFraction: {fraction:.1f} | DL Rate: {avg_dl_rate:.2f} | WMMSE: {avg_wmmse_rate:.2f} | Ratio: {ratio:.2f}")

    print("\n")


class RawOnlyPowerNet(nn.Module):
    def __init__(self, n_users, n_antennas, n_subcarriers, hidden_dim=256):
        super().__init__()
        self.n_users = n_users
        self.n_antennas = n_antennas
        
        # Input: Flattened vector for one subcarrier (all users + antennas)
        input_dim = n_users * n_antennas * 2
        
        # A simple but deep MLP to compete fairly with the "Head" of the LWM
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_users * n_antennas)
        )

    def forward(self, h_complex, p_max=1.0):
        batch_size, s, k, n, _ = h_complex.shape
        
        # Flatten input: [Batch, S, Input_Dim]
        # Note: This MLP processes each subcarrier INDEPENDENTLY.
        # It has no attention mechanism to see "other" subcarriers.
        x = h_complex.view(batch_size, s, -1)
        
        # Pass through MLP
        raw_val = self.net(x) # [Batch, S, K*N]
        
        # Global Power Constraint
        flat_val = raw_val.view(batch_size, -1)
        power_dist = F.softmax(flat_val, dim=1)
        final_flat = power_dist * p_max
        final_power = final_flat.view(batch_size, s, k, n)
        
        return final_power