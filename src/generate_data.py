from scenario_props import *
import numpy as np
import DeepMIMOv3
import torch

def get_parameters(scenario, bs_idx=1):
    n_ant_ue = 1
    scs = 30e3
      
    row_column_users = scenario_prop()
    
    parameters = DeepMIMOv3.default_params()
    parameters['dataset_folder'] = './scenarios'
    parameters['scenario'] = scenario.split("_v")[0]
    
    n_ant_bs = row_column_users[scenario]['n_ant_bs']
    n_subcarriers = row_column_users[scenario]['n_subcarriers']
    parameters['active_BS'] = np.array([bs_idx])
    
    if isinstance(row_column_users[scenario]['n_rows'], int):
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'])
    else:
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'][0],
                                            row_column_users[scenario]['n_rows'][1])

    parameters['bs_antenna']['shape'] = np.array([n_ant_bs, 1]) # Horizontal, Vertical 
    parameters['bs_antenna']['rotation'] = np.array([0,0,-135]) # (x,y,z)
    parameters['ue_antenna']['shape'] = np.array([n_ant_ue, 1])
    parameters['enable_BS2BS'] = False
    parameters['OFDM']['subcarriers'] = n_subcarriers
    parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    
    parameters['OFDM']['bandwidth'] = scs * n_subcarriers / 1e9
    parameters['num_paths'] = 20

    return parameters


def deepmimo_data_gen(scenario_name=None, bs_idxs=[1,2,3]):
    deepmimo_data = []

    for bs_idx in bs_idxs:
        parameters = get_parameters(scenario_name, bs_idx)
        deepMIMO_dataset = DeepMIMOv3.generate_data(parameters)[0]

        n_ant_bs = parameters['bs_antenna']['shape'][0]
        n_subcarriers = parameters['OFDM']['subcarriers']
        n_ant_ue = parameters['ue_antenna']['shape'][0]
    
        deepmimo_data.append(deepMIMO_dataset)

    return deepmimo_data, n_ant_ue, n_ant_bs, n_subcarriers

def deepmimo_data_cleaning(deepmimo_data):
    idxs = np.where(deepmimo_data['user']['LoS'] != -1)[0]
    cleaned_deepmimo_data = deepmimo_data['user']['channel'][idxs]
    return np.array(cleaned_deepmimo_data) * 1e6

def sample_users(dataset, M):
    idxs = np.random.randint(0, len(dataset), M)
    return dataset[idxs]

def compute_U_by_svd(H):
    """
    Calcula os combiners U via SVD (primeira coluna de U) para cada usuário e subcarrier.
    
    Parameters:
        H: np.array, shape (K, N, M, SC), complex
           Canais do BS para cada usuário (K), subcarrier (SC), RX antennas (N), TX antennas (M)
    
    Returns:
        U: np.array, shape (K, SC, N), complex
           Vetores combiners normalizados para cada usuário e subcarrier
    """
    K, N, M, SC = H.shape
    U = np.zeros((K, SC, N), dtype=complex)
    
    for k in range(K):
        for s in range(SC):
            Hks = H[k,:,:,s]  # (N, M)
            
            # SVD
            try:
                u, _, _ = np.linalg.svd(Hks, full_matrices=False)
                u1 = u[:, 0]  # primeira coluna
            except np.linalg.LinAlgError:
                # fallback caso SVD falhe
                cov = Hks @ Hks.conj().T
                vals, vecs = np.linalg.eigh(cov)
                u1 = vecs[:, -1]
            
            # normalização
            norm = np.linalg.norm(u1)
            if norm == 0:
                u1 = np.ones(N, dtype=complex) / np.sqrt(N)
            else:
                u1 = u1 / norm
            
            U[k, s, :] = u1
    
    return U

def compute_ZF_precoder(H, U):
    """
    Calcula o precoder V via Zero-Forcing usando os combiners U.
    
    Parameters:
        H: np.array, shape (K, N, M, SC), complex
        U: np.array, shape (K, SC, N), complex
    
    Returns:
        V_all: list of length SC, each array shape (M, K)
               Cada coluna V_all[s][:,k] é o precoder do usuário k no subcarrier s
    """
    K, N, M, SC = H.shape
    V_all = []
    
    for s in range(SC):
        # Monta a matriz C (K x M)
        C = np.zeros((K, M), dtype=complex)
        for k in range(K):
            uks = U[k, s, :].conj().reshape(1, -1)  # (1,N)
            Hks = H[k, :, :, s]                     # (N,M)
            C[k, :] = (uks @ Hks).reshape(M,)
        
        # ZF precoder: V = C^H * (C C^H)^{-1}
        try:
            CC_H_inv = np.linalg.inv(C @ C.conj().T)
        except np.linalg.LinAlgError:
            CC_H_inv = np.linalg.pinv(C @ C.conj().T)
        
        V = C.conj().T @ CC_H_inv  # (M,K)
        
        # Normaliza colunas
        norms = np.linalg.norm(V, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        V = V / norms
        
        V_all.append(V)
    
    return V_all

def compute_gains(H, U, V_all):
    """
    Calcula os ganhos efetivos g_{k,s} para cada usuário/subcarrier.
    
    Parameters:
        H: np.array, shape (K, SC, N, M), complex
        U: np.array, shape (K, SC, N), complex
        V_all: list de SC arrays, cada um (M, K)
    
    Returns:
        gains: np.array, shape (K, SC), float
            Ganhos efetivos |u^H H v|^2
        cross_gains: np.array, shape (K, K, SC), float
            Ganhos cruzados |u^H H v_j|^2
    """
    K, N, M, SC = H.shape
    gains = np.zeros((K, SC), dtype=float)
    cross_gains = np.zeros((K, K, SC), dtype=float)
    
    for s in range(SC):
        V = V_all[s]  # (M, K)
        for k in range(K):
            uks = U[k, s, :].conj().reshape(1, -1)  # (1, N)
            Hks = H[k, :, :, s]                     # (N, M)
            for j in range(K):
                vjs = V[:, j].reshape(-1, 1)        # (M,1)
                val = np.abs(uks @ Hks @ vjs)**2
                cross_gains[k, j, s] = val
                if j == k:
                    gains[k, s] = val
    return gains, cross_gains

def water_filling(gains, Ptot, sigma2, tol=1e-6, max_iter=1000):
    """
    Aplica water-filling para alocar potência.
    
    Parameters:
        gains: np.array, shape (K, SC), ganhos |u^H H v|^2
        Ptot: float, potência total disponível
        sigma2: float, variância do ruído
        tol: float, tolerância para convergência
        max_iter: int, limite de iterações
    
    Returns:
        p_alloc: np.array, shape (K, SC), potências alocadas
    """
    K, SC = gains.shape
    n_channels = K * SC
    inv_snr = sigma2 / (gains + 1e-12)  # evitar div zero
    
    # limites para busca binária
    mu_low = np.min(inv_snr)
    mu_high = mu_low + Ptot + 1.0
    
    for _ in range(max_iter):
        mu = 0.5 * (mu_low + mu_high)
        p_alloc_ks = np.maximum(0, mu - inv_snr)
        power_used = np.sum(p_alloc_ks)
        
        if np.abs(power_used - Ptot) < tol:
            break
        if power_used > Ptot:
            mu_high = mu
        else:
            mu_low = mu
    
    p_alloc_k = p_alloc_ks.sum(axis=1)

    return p_alloc_ks, p_alloc_k

def create_samples(scenario_name=None, bs_idxs=[1,2,3], N_samples=1000):

    deepmimo_data, n_ant_ue, n_ant_bs, n_subcarriers = deepmimo_data_gen(scenario_name, bs_idxs)
    cleaned_channels = [deepmimo_data_cleaning(deepmimo_data[bs_idx]) for bs_idx in range(len(deepmimo_data))]

    samples = []
    for i in range(N_samples):

        bs_idx = np.random.randint(0 ,len(bs_idxs))

        sample = sample_users(cleaned_channels[bs_idx], n_ant_bs)
        U = compute_U_by_svd(sample)
        V_all = compute_ZF_precoder(sample, U)
        gains, cross_gains = compute_gains(sample, U, V_all)
        _, p_alloc = water_filling(gains, 100, 10e-3)

        samples.append({"channels": sample, "p_alloc": p_alloc})

    return samples

def patch_maker(original_ch, patch_rows, patch_cols):
    n_samples, _, n_rows, n_cols = original_ch.shape

    # Setp 1: Remove Singleton dimension - Currently, the LWM model uses only one antenna in the UE
    original_ch = original_ch[:, 0]

    # Step 2: Split into real and imaginary parts and interleave them
    flat_real = original_ch.real
    flat_imag = original_ch.imag

    # Interleave real and imaginary parts along the last axis
    interleaved = np.empty((n_samples, n_rows, n_cols * 2), dtype=np.float32)
    interleaved[:, :, 0::2] = flat_real
    interleaved[:, :, 1::2] = flat_imag

    # Step 3: Compute the number of patches along rows and columns
    n_patches_rows = int(np.ceil(n_rows / patch_rows))
    n_patches_cols = int(np.ceil(n_cols / patch_cols))

    # Step 4: Pad the matrix if necessary to make it divisible by patch size
    padded_rows = n_patches_rows * patch_rows - n_rows
    padded_cols = n_patches_cols * patch_cols - n_cols
    if padded_rows > 0 or padded_cols > 0:
        interleaved = np.pad(
            interleaved,
            ((0, 0), (0, padded_rows), (0, padded_cols * 2)),  # Double padding for interleaved axis
            mode='constant',
            constant_values=0,
        )

    # Step 5: Create patches by dividing into blocks
    n_samples, padded_rows, padded_cols = interleaved.shape
    padded_cols //= 2  # Adjust for interleaving (real and imaginary parts count as one)
    patches = []

    for i in range(0, padded_rows, patch_rows):
        for j in range(0, padded_cols, patch_cols):
            patch = interleaved[:, i:i + patch_rows, j * 2:(j + patch_cols) * 2]
            patches.append(patch.reshape(n_samples, -1))  # Flatten each patch

    # Step 6: Stack patches to form the final array
    patches = np.stack(patches, axis=1)  # Shape: (num_samples, n_patches, patch_rows * patch_cols * 2)

    return patches
