import torch
import numpy as np

EPS = 1e-12

def zf_precoder(H, P_max, sigma2=1.0):
    """
    Calculates Zero-Forcing Beamforming and Sum Rate.
    
    Args:
        H: Channel (N, K, M)
        P_max: Total power constraint (scalar)
        sigma2: Noise variance (scalar)
        
    Returns:
        V: Precoding matrix (N, M, K)
        sum_rate: Achieved sum rate (scalar)
    """
    N, K, M = H.shape

    # Sanity check: ZF requires at least as many antennas as users
    if M < K:
        raise ValueError(f"Zero-Forcing requires M >= K. Got M={M}, K={K}.")

    # 1. Calculate Moore-Penrose Pseudo-inverse
    # H is (N, K, M) -> pinv(H) is (N, M, K)
    V_raw = np.linalg.pinv(H)

    # 2. Power Normalization (Total Power Constraint)
    # We simply scale the raw ZF matrix so the total power equals P_max
    current_total_power = np.sum(np.abs(V_raw)**2)
    
    # Scaling factor alpha
    alpha = np.sqrt(P_max / current_total_power)
    
    V = V_raw * alpha

    # 3. Calculate Sum Rate
    # We calculate the exact SINR to account for numerical residuals
    
    # Effective Channel: (N, K, M) @ (N, M, K) -> (N, K, K)
    # Ideally, this is a diagonal matrix (Interference is 0)
    HV = H @ V 
    
    # Signal Power: Magnitude squared of diagonal elements
    signal_power = np.abs(np.einsum('nkk->nk', HV))**2
    
    # Interference: Sum of squares of off-diagonals
    # Total power received per user - Signal Power
    total_rx_power = np.sum(np.abs(HV)**2, axis=2)
    interference_power = total_rx_power - signal_power
    
    # Note: For ZF, interference_power should be ~1e-30 (machine epsilon)
    # But we include it for mathematical correctness
    sinr = signal_power / (interference_power + sigma2)
    
    sum_rate = np.sum(np.log2(1 + sinr))

    return V, sum_rate

def update_u(H, V, sigma2=1.0):
    """
    Computes MMSE receiver scalar U for all N subcarriers simultaneously.
    H: (N, K, M)
    V: (N, M, K)
    """
    # 1. Batch Matrix Multiplication
    # (N, K, M) @ (N, M, K) -> (N, K, K)
    # Contains signal (diagonal) and interference (off-diagonal) for all N
    HV = H @ V 

    # 2. Calculate Total Received Power (Denominator)
    # We sum across axis 2 (the streams j) for each user k (axis 1)
    # Shape: (N, K, K) -> (N, K)
    total_power = np.sum(np.abs(HV)**2, axis=2) + sigma2

    # 3. Extract Desired Signal (Numerator)
    # We need the diagonal elements of the last two dimensions
    # einsum 'nkk->nk' grabs the diagonal k,k for every batch n
    effective_gain = np.einsum('nkk->nk', HV)

    # 4. MMSE Update
    U = np.conjugate(effective_gain) / total_power

    return U

def update_w(H, V, U):
    """
    H: (N, K, M) - Channels
    V: (N, M, K) - Precoders
    U: (N, K)    - Receiver scalars
    """
    # 1. Efficiently compute ONLY the diagonal elements of H @ V
    # We want to sum over dimension M for matching users K
    # Einstein Summation: 
    # n: batch (subcarriers)
    # k: user index (row in H)
    # m: antenna index (contraction dim)
    # k: user index (col in V, corresponds to user k's stream)
    # -> nk: output is (N, K) scalar signals
    
    signal_gain = np.einsum('nkm,nmk->nk', H, V) # Returns complex (N, K)

    # 2. Calculate MSE (1 - u * h * v)
    # Note: U is usually conjugated in the formula, but in code U is often 
    # already stored as the conjugate value from the update_u step. 
    # Assuming your U input is already the correct conjugate factor:
    mse_term = 1.0 - np.real(U * signal_gain)

    # 3. Compute W with stability protection
    # Clip minimum MSE to 1e-6 to prevent division by zero/huge weights
    W = 1.0 / np.maximum(mse_term, 1e-6)

    return W

def update_v(H, U, W, P_total):
    """
    Optimizes V for all subcarriers under a GLOBAL Total Power Constraint.
    N: Subcarriers, K: Users, M: Antennas
    """
    N, K, M = H.shape
    
    # --- 1. Construct A and B matrices for all subcarriers (Vectorized) ---
    # We want to build A_n and B_n simultaneously for all n=0..N-1
    
    # Weights: (N, K)
    weights = W * np.abs(U)**2
    
    # Reshape H for broadcasting: (N, K, M, 1)
    H_r = H.reshape(N, K, M, 1)
    
    # Calculate outer products h_k^H * h_k: (N, K, M, M)
    # (N, K, M, 1) @ (N, K, 1, M) -> (N, K, M, M)
    HH = np.matmul(H_r.conj(), H_r.transpose(0, 1, 3, 2))
    
    # A: Weighted sum of outer products over K users -> (N, M, M)
    # Sum(weights * HH)
    A = np.sum(weights[:, :, None, None] * HH, axis=1)

    # B: Term w_k * u_k^* * h_k^H -> (N, M, K)
    # We need B[:, :, k] to be the vector for user k
    # w*u_conj: (N, K) -> broadcast
    # H_hermitian: (N, K, 1, M) -> transpose to get h^H
    factors = (W * np.conj(U))[:, :, None, None] # (N, K, 1, 1)
    H_herm = H_r.transpose(0, 1, 3, 2).conj()    # (N, K, 1, M)
    B_raw = factors * H_herm                      # (N, K, 1, M)
    B = B_raw.squeeze(2).transpose(0, 2, 1)       # (N, M, K)

    # --- 2. The EVD Trick (Crucial for Performance) ---
    # Since A is Hermitian, diagonalize it: A = Q * Lambda * Q^H
    # This is done ONCE per update_v call, outside the bisection loop.
    lambdas, Q = np.linalg.eigh(A) # lambdas: (N, M), Q: (N, M, M)

    # Transform B into the eigenspace: B_tilde = Q^H * B
    # (N, M, M).conj().T is tricky in numpy, use explicit transpose swap
    Q_H = Q.transpose(0, 2, 1).conj()
    B_tilde = np.matmul(Q_H, B) # (N, M, K)

    # --- 3. Global Bisection Search for mu ---
    # We look for one mu such that Sum_N(Power_n(mu)) == P_total
    
    def calc_total_power(mu):
        # The inverse of (Lambda + mu*I) is just 1/(lambda + mu)
        # We apply this diagonal scaling to B_tilde
        # Denominator: (N, M, 1) to broadcast over K streams
        inv_diag = 1.0 / (lambdas + mu)[:, :, None] 
        
        # V_tilde = (Lambda + mu*I)^-1 * B_tilde
        V_tilde = inv_diag * B_tilde 
        
        # Power is invariant under unitary transformation Q, 
        # so sum(|V|^2) == sum(|V_tilde|^2). We don't need to rotate back yet!
        return np.sum(np.abs(V_tilde)**2)

    # Check unconstrained power (mu=0)
    if calc_total_power(0.0) <= P_total:
        opt_mu = 0.0
    else:
        # Find mu such that power == P_total
        low, high = 0.0, 1e5 # Adjust upper bound if needed
        for _ in range(20): # 20 iters is usually enough for float32 precision
            mid = (low + high) / 2
            p_val = calc_total_power(mid)
            if p_val > P_total:
                low = mid
            else:
                high = mid
        opt_mu = high

    # --- 4. Reconstruct Final V ---
    # Apply optimal mu, rotate back with Q
    inv_diag_opt = 1.0 / (lambdas + opt_mu)[:, :, None]
    V_tilde_opt = inv_diag_opt * B_tilde
    V = np.matmul(Q, V_tilde_opt) # (N, M, K)

    return V

def wmmse(H, P_max, sigma2=1.0):
    N, K, M = H.shape
    
    # --- 1. Initialization with Power Normalization ---
    V, _ = zf_precoder(H, P_max, sigma2)
    
    # Calculate current random power
    # (Assuming Total Power Constraint over all N subcarriers)
    current_power = np.sum(np.abs(V)**2)
    
    # Scale V to exactly match P_max initially
    V = V * np.sqrt(P_max / current_power)

    sum_rate = 0.0
    
    # --- 2. Optimization Loop ---
    for i in range(200):
        # Update order: U -> W -> V
        U = update_u(H, V, sigma2)
        W = update_w(H, V, U)
        # Ensure you use the Corrected/Vectorized update_v from previous advice
        V = update_v(H, U, W, P_max) 
        
        # Convergence check using the W-proxy (computationaly cheap)
        # We take abs(W) just to be safe, though W should be real positive
        current_sum_rate_proxy = np.sum(np.log2(np.abs(W)))
        
        if abs(sum_rate - current_sum_rate_proxy) < 1e-4:
            break
            
        sum_rate = current_sum_rate_proxy

    # --- 3. Final Accurate Rate Calculation ---
    # Recalculate exact SINR one last time to ensure the returned rate is real physics
    # (Re-using code logic for SINR calculation)
    HV = H @ V
    signal_power = np.abs(np.einsum('nkk->nk', HV))**2
    inter_plus_noise = np.sum(np.abs(HV)**2, axis=2) - signal_power + sigma2
    sinr = signal_power / inter_plus_noise
    final_exact_rate = np.sum(np.log2(1 + sinr))

    return V, final_exact_rate

def sum_rate_loss(power_alloc, channel_matrix, noise_var=1.0):
    """
    Calculates Sum Rate considering Inter-User Interference (SINR).
    Assumption: BS uses MRT Phases + AI-Predicted Amplitudes.
    
    Args:
        power_alloc: [Batch, S, K, N] (Power magnitudes per antenna)
        channel_matrix: [Batch, S, K, N, 2] (Raw channel Real/Imag)
    """
    epsilon = 1e-9

    # 1. Convert Raw Channel to Complex Tensor [Batch, S, K_rx, N_ant]
    h_complex = torch.view_as_complex(channel_matrix)
    
    # 2. Construct the Beamformer (Precoder) Vectors
    # MRT Phase (Conjugate Beamforming) + AI Amplitude
    w_mag = torch.sqrt(power_alloc + epsilon)
    w_phase = -h_complex.angle()
    w_complex = w_mag * torch.exp(1j * w_phase)

    # 3. Compute Received Signal Matrix (Interference Map)
    # Interaction G_ij = h_i . w_j
    # Einsum: b=batch, s=subcarrier, i=Rx User, j=Tx User, n=Antenna
    interaction_matrix = torch.einsum('bsin,bsjn->bsij', h_complex, w_complex)
    
    # 4. Compute Power Received
    power_received_matrix = (interaction_matrix.abs()) ** 2
    
    # 5. Extract Signal and Interference
    # Diagonal (i==j) is Signal
    signal_power = torch.diagonal(power_received_matrix, dim1=-2, dim2=-1) 
    # Sum over Tx users (j) is Total Power
    total_power_received = torch.sum(power_received_matrix, dim=-1)        
    # Interference = Total - Signal
    interference_power = total_power_received - signal_power
    
    # 6. SINR Calculation
    sinr = signal_power / (interference_power + noise_var + epsilon)
    
    # 7. Rate Calculation (Shannon)
    rate_per_subcarrier = torch.log2(1 + sinr)
    total_sum_rate = torch.sum(rate_per_subcarrier, dim=(1, 2))
    
    return -torch.mean(total_sum_rate)