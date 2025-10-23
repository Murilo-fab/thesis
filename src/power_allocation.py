import numpy as np

def mmse_precoder(H, P_tot, sigma2):
    M, K = H.shape
    
    alpha = K*sigma2/P_tot
    # A = H H^H + alpha I_M
    A = H @ H.conj().T + alpha * np.eye(M)
    # U = A^-1 H
    U = np.linalg.solve(A, H)
    # Normatization
    tr = np.trace(U @ U.conj().T)
    beta = np.sqrt(P_tot/tr)
    W = beta*U

    return W

def beamforming(W):
    # W: (M, K) precoder MMSE já calculado
    M, K = W.shape

    # Inicializa arrays
    F = np.zeros_like(W, dtype=np.complex128)  # direções unitárias
    p_init = np.zeros(K)                        # potências iniciais

    for k in range(K):
        norm_wk = np.linalg.norm(W[:, k])
        if norm_wk > 0:
            F[:, k] = W[:, k] / norm_wk
            p_init[k] = norm_wk**2
        else:
            F[:, k] = np.zeros(M)  # caso w_k = 0
            p_init[k] = 0.0
    
    return F

def power_allocation():
    pass

def power_allocation_wmmse_fixed_dirs(H, F, P_tot, sigma2,
                                      max_iter=200, tol=1e-6, eps=1e-12):
    """
    Optimiza potências p_k >=0 com direções fixas F (colunas unit-norm)
    para maximizar soma de taxas aproximada pelo algoritmo WMMSE.
    Entradas:
      H: (M, K) canais, col k = h_k (complex)
      F: (M, K) direções normalizadas, col k = f_k (complex), ||f_k||=1
      P_tot: potência total disponível (float)
      sigma2: ruído (float)
      max_iter, tol: controle de iteração
    Retorna:
      p: vetor de potências ótimas (K,)
      rates: soma das taxas ao final (bits/s/Hz)
      history: dict com histórico de soma-rate por iteração
    """
    M, K = H.shape
    # pré-calcula os ganhos escalares g_kj = h_k^H f_j -> matriz KxK
    G = np.zeros((K, K), dtype=np.complex128)
    for k in range(K):
        G[k, :] = H[:, k].conj().T @ F  # row k contains g_kj for j=1..K

    p = np.ones(K) * (P_tot / K)

    history = {'sumrate': []}

    for it in range(max_iter):
        # 1) compute interference+noise for each user
        int_plus_noise = (np.abs(G)**2) @ p + sigma2  # shape (K,)
        # 2) compute u_k (scalar MMSE receive)
        # note: sqrt(p_k) * g_kk is complex; u_k is complex
        u = (np.sqrt(p) * np.diag(G).conj()) / int_plus_noise
        # 3) compute MSE_k
        # term1 = 1
        # term2 = -2 Re{u_k^* sqrt(p_k) g_kk}
        # term3 = |u_k|^2 * (sum_j p_j |g_kj|^2 + sigma2)
        term2 = -2.0 * np.real(u.conj() * (np.sqrt(p) * np.diag(G)))
        term3 = (np.abs(u)**2) * int_plus_noise
        MSE = 1.0 + term2 + term3
        # avoid division by zero
        MSE = np.maximum(MSE, eps)
        # 4) weights
        w = 1.0 / MSE  # shape (K,)

        # 5) compute c_j and d_j
        # c_j = sum_k w_k |u_k|^2 |g_kj|^2   -> shape (K,)
        # d_j = w_j * Re{u_j^* g_jj}
        absG2 = np.abs(G)**2  # KxK
        c = (w * (np.abs(u)**2)) @ absG2  # shape (K,)
        d = w * np.real(u.conj() * np.diag(G))  # shape (K,)
        # force nonnegative d
        d = np.maximum(d, 0.0)

        # 6) find lambda >=0 s.t. sum_j (d_j/(c_j+lambda))^2 = P_tot
        # if c_j negative numerical (shouldn't), clip to >=0
        c = np.maximum(c, 0.0)
        # closed form if lambda = 0 satisfies power
        def compute_power_sum(lmbda):
            t = d / (c + lmbda)
            return np.sum(t**2)

        # check if lambda=0 satisfies constraint
        if compute_power_sum(0.0) <= P_tot:
            lam = 0.0
        else:
            # bissecção para lambda
            lam_lo = 0.0
            lam_hi = 1.0
            # find hi such that power_sum(hi) < P_tot
            while compute_power_sum(lam_hi) > P_tot:
                lam_hi *= 2.0
                if lam_hi > 1e12:
                    break
            # bissecção
            for _ in range(60):
                lam_mid = 0.5 * (lam_lo + lam_hi)
                if compute_power_sum(lam_mid) > P_tot:
                    lam_lo = lam_mid
                else:
                    lam_hi = lam_mid
            lam = 0.5*(lam_lo + lam_hi)

        t = d / (c + lam)
        p_new = t**2
        # project tiny negatives to zero (numerical)
        p_new = np.maximum(p_new, 0.0)

        # compute sum-rate for monitoring
        sinr = (p_new * np.abs(np.diag(G))**2) / ( (np.abs(G)**2) @ p_new - p_new * np.abs(np.diag(G))**2 + sigma2 )
        rates = np.log2(1.0 + np.maximum(sinr, 0.0))
        sumrate = np.sum(rates)
        history['sumrate'].append(sumrate)

        # stop criterion
        if np.linalg.norm(p_new - p) / (np.linalg.norm(p) + 1e-12) < tol:
            p = p_new
            break
        p = p_new

    return p, sumrate, history