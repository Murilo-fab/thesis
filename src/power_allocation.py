import torch

epsilon = 1e-18

def solve_water_filling(gains, P_total=1.0, noise_variance=1.0):
    """
    Solver for Water Filling
    
    Inputs:
        gains (torch.tensor): Gains tensor [B, S, K]
        P_total (float): Total power for the system

    Output:
        power (torch.tensor): Power tensor [B, S, K]
    """
    # 1. Flatten dimensions 
    # Treat all users and subcarriers as a list of independent parallel channels.
    # Shape: (B, S, K) : (B, S*K)
    B, S, K = gains.shape
    flat_gains = gains.view(B, S * K)

    # 2. Clamps small gains to increase numerical stability
    flat_gains = torch.clamp(flat_gains, min=epsilon)

    # 3. Calculate noise levels
    noise_levels = noise_variance / flat_gains

    # 4. Sort noise levels from low to high / good to bad channels
    sorted_noise, indices = torch.sort(noise_levels, dim=-1)

    # 5. Calculate the cumulative sum of noises for k=1, k=2, k=3...
    cum_noise = torch.cumsum(sorted_noise, dim=-1)

    num_channels = S * K
    k_range = torch.arange(1, num_channels + 1, device=gains.device).view((1, num_channels))

    # 6. Calculates the water level mu for every possible number of active channels
    mu_candidates = (P_total + cum_noise) / k_range

    # 7. Check if the water level is greater than the noise floor for the k-th user
    valid = mu_candidates > sorted_noise

    # 8. Finds the max k that satisfy the criterion
    # We clamp min=1 to prevent k_optimal being 0. 
    # This prevents the index from ever being -1.
    k_optimal = torch.sum(valid, dim=-1, keepdim=True).clamp(min=1)
    
    # 9. Selects the correct mu
    k_idx = k_optimal - 1
    mu_selected = torch.gather(mu_candidates, -1, k_idx)

    # 10. Calculates power: P = mu - noise
    sorted_powers = torch.clamp(mu_selected - sorted_noise, min=0)

    # 11. Restore the original order
    flat_powers = torch.zeros_like(flat_gains)
    flat_powers.scatter_(-1, indices, sorted_powers)

    powers = flat_powers.view(B, S, K)

    return powers

def apply_water_filling(W, H, P_total=1.0, noise_variance=1.0):
    """
    Apply Water Filling to optimize the power allocation for a precoder
    
    Inputs:
        W (torch.tensor): Initial precoder [B, S, N, K]
        H (torch.tensor): Channel tensor [B, S, K, N]
        P_total (float): Total power for the system

    Output:
        W (torch.tensor): Refined precoder with optimal power [B, S, N, K]
    """

    # 1. Extract directions
    # Normalize the columns of W to unit form
    V_directions = torch.nn.functional.normalize(W, p=2, dim=-2, eps=epsilon)

    # 2. Calculate "effective channel" with those directions (H @ V)
    # Shape: (B, S, K, N) @ (B, S, N, K) : (B, S, K, K)
    H_eff = H @ V_directions

    # 3. Get the gains from the "effective channel"
    signal_amplitudes = torch.diagonal(H_eff, dim1=-2, dim2=-1)
    effective_gains = torch.abs(signal_amplitudes)**2

    # 4. Solve Water Filling 
    optimal_powers = solve_water_filling(effective_gains, P_total, noise_variance)

    # 5. Apply new powers
    W_opt = V_directions * torch.sqrt(optimal_powers.unsqueeze(-2))

    return W_opt

def mrt_precoder(H: torch.tensor, P_total=1.0):
    """
    Normalized MRT Precoder for a Multi-Carrier MU-MIMO system.
    
    Inputs:
        H (torch.tensor): Channel tensor [B, S, K, N]
        P_total (float): Total power for the system

    Output:
        W (torch.tensor): Normalized Precoder [B, S, N, K]
    """

    # 1. MRT precoder is the conjugate transpose of H
    W_raw = H.transpose(-2, -1).conj()

    # 2. Normalize
    mag_squared = torch.abs(W_raw)**2
    system_energy = torch.sum(mag_squared, dim=(-3, -2, -1), keepdim=True)
    scaling_factor = torch.sqrt(P_total / system_energy)

    W_norm = W_raw * scaling_factor

    return W_norm

def get_mrt_directions(H):
    """
    Returns the Unit-Norm MRT Beamforming Directions
    
    Inputs:
        H (torch.tensor): Channel matrix [B, S, K, N]

    Outputs:
        V (torch.tensor) : Normalized directions [B, S, N, K]
    """
    # 1. Conjugate Transpose (Simple MRT)
    # Shape: [B, S, N, K]
    W_raw = H.transpose(-2, -1).conj()
    
    # 2. Normalize columns to length 1.0 (Pure Direction)
    V_directions = torch.nn.functional.normalize(W_raw, p=2, dim=-2)
    
    return V_directions

def zf_precoder(H: torch.tensor, P_total=1.0):
    """
    Calculates the Normalized ZF Precoder for a Multi-Carrier MU-MIMO system.

    Inputs:
        H (torch.tensor): Channel tensor [B, S, K, N]
        P_total (float): Total power for the system

    Output:
        W (torch.tensor): Normalized Precoder [B, S, N, K]
    """

    # 1. Hermetian Transpose
    H_hermitian = H.transpose(-2, -1).conj()

    # 2. Gram Matrix
    # Shape: (B, S, K, N) @ (B, S, N, K) : (B, S, K, K)
    gram_matrix = H @ H_hermitian

    # 3. Inverse
    # For numerical stability, we add epsilon
    eye = torch.eye(gram_matrix.shape[-1], device=H.device, dtype=H.dtype)
    gram_matrix = gram_matrix + eye * epsilon

    gram_inverse = torch.linalg.inv(gram_matrix)

    # 4. Raw precoder
    # Shape: (B, S, N, K) @ (B, S, K, K) : (B, S, N, K)
    W_raw = H_hermitian @ gram_inverse

    # 5. Normalization
    mag_squared = torch.abs(W_raw)**2
    system_energy = torch.sum(mag_squared, dim=(-3, -2, -1), keepdim=True)
    scaling_factor = torch.sqrt(P_total / system_energy)

    W_norm = W_raw * scaling_factor

    return W_norm

def get_zf_directions(H: torch.tensor):
    """
    Returns the Unit-Norm ZF Beamforming Directions
    
    Inputs:
        H (torch.tensor): Channel matrix [B, S, K, N]

    Outputs:
        V (torch.tensor) : Normalized directions [B, S, N, K]
    """
    # 1. ZF calculation
    H_hermitian = H.transpose(-2, -1).conj()

    # 1.2 Inverse gram matrix
    gram = H @ H_hermitian
    eye = torch.eye(gram.shape[-1], device=H.device, dtype=H.dtype)
    gram_inv = torch.linalg.inv(gram + eye * epsilon)

    # 1.3 ZF Precoder
    W_raw = H_hermitian @ gram_inv

    # 2. Get ZF directions
    V_directions = torch.nn.functional.normalize(W_raw, p=2, dim=-2)

    return V_directions

import torch

def wmmse_solver(H,
                       P_total: float = 1.0,
                       noise_variance: float = 1e-10,
                       iterations: int = 20):
    """
    WMMSE Solver with Global Power Constraint across Subcarriers.
    
    Inputs:
        H (torch.Tensor): Channel Matrix Shape: [B, SC, K, N]
        P_total (flaot): Total system power budget
        noise_variance (float): Sigma^2
        iterations (int)
    """
    # 1. Setup
    # Ensure shape is [B, S, K, M]
    if H.dim() == 3: # [B, K, M] -> Single Carrier
        H = H.unsqueeze(1)
        
    B, S, K, M = H.shape
    device = H.device
    dtype = H.dtype

    # 2. Random Initialization
    # Random V: [B, S, N, K]
    V = torch.randn(B, S, M, K, dtype=dtype, device=device)
    
    # Scale initial V to meet P_total exactly
    # Calculate total power per batch sample (sum over S, M, K)
    sys_power = torch.sum(torch.abs(V)**2, dim=(1, 2, 3), keepdim=True)
    V = V * torch.sqrt(P_total / sys_power)
    
    # Identity matrix for regularization [1, 1, M, M]
    I = torch.eye(M, device=device).view(1, 1, M, M)

    # 3. Iteration loop
    for it in range(iterations):
        
        # Step A: Receiver (U)
        # H: [B, S, K, N], V: [B, S, N, K]
        # HV: [B, S, K, K] -> Signal at Rx k from Tx j
        HV = torch.matmul(H, V)
        
        # Signal Power (Diagonal)
        Signal = torch.diagonal(HV, dim1=2, dim2=3) # [B, S, K]
        
        # Total Rx Power (Sum over transmitters) + Noise
        Rx_Power = torch.sum(torch.abs(HV)**2, dim=3) + noise_variance # [B, S, K]
        
        # MMSE Receiver U: [B, S, K]
        U = Signal.conj() / Rx_Power
        
        # Step B: Weights (W)
        # MSE = 1 - u * signal
        MSE = 1.0 - (U.conj() * Signal).real
        MSE = torch.clamp(MSE, min=1e-6)
        W = 1.0 / MSE # [B, S, K]

        # Step C: Transmitter (V)
        # Solve (A_s + mu * I)^-1 * B_s for every subcarrier s
        # But mu is SHARED across s.
        
        # 1. Construct A_s (Covariance) per subcarrier
        # scaler: sqrt(w) * |u|
        scaler_A = torch.sqrt(W) * torch.abs(U) # [B, S, K]
        H_scaled = H * scaler_A.unsqueeze(3)    # [B, S, K, M]
        # A = H_scaled^H @ H_scaled -> [B, S, M, M]
        A = torch.matmul(H_scaled.transpose(2, 3).conj(), H_scaled)
        
        # 2. Construct B_s (Target) per subcarrier
        scaler_B = W * U # [B, S, K]
        # B_target = H^H * diag(scaler) -> [B, S, M, K]
        B_target = H.transpose(2, 3).conj() * scaler_B.unsqueeze(2)

        # Global biscetion search for MU
        # We search for a single scalar mu per batch item
        
        # Initial Bounds: [B, 1, 1, 1] for broadcasting
        mu_min = torch.zeros(B, 1, 1, 1, device=device)
        mu_max = torch.ones(B, 1, 1, 1, device=device) * 1000.0
        
        # 1. Check Unconstrained (mu=0)
        # Solve A * V = B for all S concurrently
        # A + epsilon*I to avoid singularity
        V_try = torch.linalg.solve(A + epsilon*I, B_target)
        
        # Calculate SYSTEM Power: Sum over Subcarriers(1), Antennas(2), Users(3)
        P_sys = torch.sum(torch.abs(V_try)**2, dim=(1, 2, 3), keepdim=True) # [B, 1, 1, 1]
        
        # Identify violators
        mask_violation = (P_sys > P_total).float() # [B, 1, 1, 1]
        
        # 2. Bisection Loop (only affects violators due to masking later)
        for _ in range(20):
            mu = (mu_min + mu_max) / 2
            
            # Apply shared mu to all subcarriers
            A_reg = A + mu * I # [B, S, M, M] + [B, 1, M, M] -> [B, S, M, M]
            
            V_try = torch.linalg.solve(A_reg, B_target)
            
            # Check Global Power
            P_current = torch.sum(torch.abs(V_try)**2, dim=(1, 2, 3), keepdim=True)
            
            # Update bounds
            high_power = (P_current > P_total).float()
            mu_min = high_power * mu + (1 - high_power) * mu_min
            mu_max = high_power * mu_max + (1 - high_power) * mu
            
        # 3. Select Final V
        # If unconstrained was valid, use it. Otherwise use bisection result.
        # Recalculate V_try one last time with refined mu for violators
        # (Or just use the last V_try from loop, which is close enough)
        
        # Re-solve one last time with the "best" mu found (mixed with 0 for non-violators)
        mu_final = mask_violation * mu + (1 - mask_violation) * 0.0
        A_final = A + mu_final * I
        V_final = torch.linalg.solve(A_final, B_target)
        
        # Final safety clamp
        # Ensure strict compliance with P_total
        P_final_sys = torch.sum(torch.abs(V_final)**2, dim=(1, 2, 3), keepdim=True)
        scale_factor = torch.min(torch.ones_like(P_final_sys), torch.sqrt(P_total / (P_final_sys + 1e-12)))
        V = V_final * scale_factor

    # 4. Return rates
    HV = torch.matmul(H, V)
    Signal_Power = torch.abs(torch.diagonal(HV, dim1=2, dim2=3))**2
    Interference_Power = torch.sum(torch.abs(HV)**2, dim=3) - Signal_Power
    SINR = Signal_Power / (Interference_Power + noise_variance)
    
    # Rate per subcarrier: [B, S, K]
    Rate_SC = torch.log2(1 + SINR)
    
    Sum_Rate = torch.sum(torch.mean(Rate_SC, dim=1), dim=1)
        
    return Sum_Rate, V

def calculate_sum_rate(H, W, noise_variance=1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the Sum Rate for a precoder.
    
    Inputs:
        H (torch.tensor): Channel tensor [B, S, K, N]
        W (torch.tensor): Precoder tensor [B, S, N, K]
        noise_variance (float): Noise variance (sigma^2)

    Outputs:
        sum_rate (torch.tensor): Sum-rate per subcarrier [B, S]
        user_rates (torch.tensor): Sum-rate per user [B, S, K]
    """

    # 1. Calculate the effective channel (H @ W)
    # Shape: (B, S, K, N) @ (B, S, N, K) : (B, S, K, K)
    H_eff = H @ W

    # 2. Extract signal power (Diagonal of the effective channel matrix)
    signal_power = torch.abs(torch.diagonal(H_eff, dim1=-2, dim2=-1))**2

    # 3. Extract the total received power (Signal + interference)
    total_received_power = torch.sum(torch.abs(H_eff)**2, dim=-1)

    # 4. Isolate interference
    interference_power = total_received_power - signal_power

    # 5. Calculate SINR (Signal to Interference plus Noise Ratio)
    sinr = signal_power / (interference_power + noise_variance + epsilon)

    # 6. Calculate rate
    user_rates = torch.log2(1 + sinr + epsilon)
    sum_rate = torch.sum(user_rates, dim=-1)

    return sum_rate, user_rates
