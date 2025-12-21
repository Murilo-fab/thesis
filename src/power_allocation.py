import torch
import numpy as np

epsilon = 1e-8

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
    flat_gains = torch.clamp(flat_gains, min=1e-12)

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
    V_directions = torch.nn.functional.normalize(W, p=2, dim=-2, eps=1e-12)

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

def calculate_sum_rate(H, W, noise_variance=1.0):
    """
    Calculate the Sum Rate for a precoder.
    
    Inputs:
    H (torch.tensor): Channel tensor [B, S, K, N]
    W (torch.tensor): Precoder tensor [B, S, N, K]
    noise_variance (float): Noise variance (sigma^2)

    sum_rate (torch.tensor): Sum-rate per subcarrier [B, S]
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

    return sum_rate
