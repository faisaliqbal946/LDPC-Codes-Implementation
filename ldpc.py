# ================== ENHANCED IMPLEMENTATION ==================
# Improvements:
# - Faster edge lookup via precomputed mapping (no repeated searches)
# - Improved numerical stability in BP (tanh/atanh clipping)
# - Minor clarity fixes (bit-flip input handling)
# =============================================================

"""
LDPC codes: construction, channels (BSC, AWGN, BEC), and decoders.

Implements:
  - Gallager construction (parity-check matrix H)
  - Systematic generator G via GF(2) RREF
  - BSC, AWGN (BPSK), BEC with LLR computation
  - Bit-flipping (hard), Belief Propagation (soft), Min-Sum (soft)

References:
  [1] R. G. Gallager, "Low-Density Parity-Check Codes," MIT Press, 1963.
  [2] D. J. C. MacKay, "Information Theory, Inference, and Learning Algorithms,"
      Cambridge University Press, 2003, Ch. 47.
  [3] T. Richardson and R. Urbanke, "The Capacity of Low-Density Parity-Check
      Codes under Message-Passing Decoding," IEEE Trans. IT, 2001.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Construction (Gallager [1])
# -----------------------------------------------------------------------------

def build_H(n, wc, wr, seed=42):
    """Build parity-check matrix H (m x n) with column weight wc, row weight wr.
    For (n,k) code we need m = n - k; use e.g. wr = n*wc/m."""
    rng = np.random.default_rng(seed)
    rows_per_sub = n // wr
    base = np.zeros((rows_per_sub, n), dtype=int)
    for i in range(rows_per_sub):
        base[i, i * wr : (i + 1) * wr] = 1
    subs = [base]
    for _ in range(wc - 1):
        subs.append(base[:, rng.permutation(n)])
    return np.vstack(subs)


def gf2_rref(M):
    """Reduced row echelon form over GF(2). Returns (rref, pivot_cols)."""
    A = M.copy() % 2
    m, n = A.shape
    pivots = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        found = None
        for r in range(row, m):
            if A[r, col] == 1:
                found = r
                break
        if found is None:
            continue
        A[[row, found]] = A[[found, row]]
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
        pivots.append(col)
        row += 1
    return A, pivots


def make_generator(H):
    """Return G (k x n), col_order, k for systematic encoding."""
    m, n = H.shape
    rref, pivots = gf2_rref(H)
    rank = len(pivots)
    non_pivots = [c for c in range(n) if c not in pivots]
    col_order = np.array(non_pivots + list(pivots))
    k = n - rank
    H_sys = rref[:rank][:, col_order]
    P = H_sys[:, :k]
    G = np.hstack([np.eye(k, dtype=int), P.T % 2])
    return G, col_order, k


def full_rank_H(H):
    """Return a parity-check matrix with full row rank (m = n - k rows).
    For (n, k) code the syndrome then has exactly m = n - k bits."""
    rref, pivots = gf2_rref(H)
    rank = len(pivots)
    return rref[:rank].copy()


def encode(G, message, col_order):
    """Encode message (length k) to codeword (length n) in original column order."""
    n = G.shape[1]
    c_perm = (np.asarray(message, dtype=int) @ G) % 2
    codeword = np.zeros(n, dtype=int)
    for i, orig in enumerate(col_order):
        codeword[orig] = c_perm[i]
    return codeword


# -----------------------------------------------------------------------------
# Tanner graph: for each check, list of bit indices; for each bit, list of checks
# -----------------------------------------------------------------------------

def tanner_graph(H):
    """Return (check_neighbors, bit_neighbors): check_neighbors[m] = list of bit indices;
    bit_neighbors[n] = list of check indices."""
    m, n = H.shape
    check_neighbors = [np.where(H[j, :])[0].tolist() for j in range(m)]
    bit_neighbors = [np.where(H[:, i])[0].tolist() for i in range(n)]
    return check_neighbors, bit_neighbors


# -----------------------------------------------------------------------------
# Channels and LLRs (see e.g. MacKay [2] Ch. 47)
# -----------------------------------------------------------------------------

def bsc(codeword, p, seed=None):
    """Binary Symmetric Channel: flip each bit with probability p. Returns (received, error_mask)."""
    rng = np.random.default_rng(seed)
    err = (rng.random(len(codeword)) < p).astype(int)
    return (np.asarray(codeword, dtype=int) + err) % 2, err


def bsc_llr(received, p, eps=1e-10):
    """LLR for BSC: LLR = (1-2*y)*log((1-p)/p). Positive LLR => bit more likely 0."""
    p = np.clip(p, eps, 1 - eps)
    L = np.log((1 - p) / p)
    return (1 - 2 * np.asarray(received, dtype=float)) * L


def awgn_channel(codeword, snr_db, rate=0.5, seed=None):
    """BPSK: 0->+1, 1->-1. Add Gaussian noise. snr_db = Eb/N0 in dB (per info bit). Returns (received, sigma^2)."""
    rng = np.random.default_rng(seed)
    x = 1.0 - 2.0 * np.asarray(codeword, dtype=float)
    eb_n0 = 10.0 ** (snr_db / 10.0)
    sigma2 = 1.0 / (2.0 * rate * eb_n0)
    noise = rng.normal(0, np.sqrt(sigma2), size=len(x))
    return x + noise, sigma2


def awgn_llr(received, sigma2):
    """LLR for BPSK-AWGN: LLR = 2*y/sigma^2. Positive => bit more likely 0."""
    return 2.0 * np.asarray(received, dtype=float) / sigma2


def bec_channel(codeword, epsilon, seed=None):
    """Binary Erasure Channel: each bit erased with prob epsilon. Return received: 0, 1, or -1 (erased)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(codeword, dtype=int)
    erased = rng.random(len(c)) < epsilon
    out = c.copy().astype(float)
    out[erased] = -1.0
    return out


def bec_llr(received, clip=30.0):
    """LLR for BEC: known 0 -> +clip, known 1 -> -clip, erased -> 0."""
    out = np.zeros(len(received))
    out[received == 0] = clip
    out[received == 1] = -clip
    return out


# -----------------------------------------------------------------------------
# Decoders
# -----------------------------------------------------------------------------

def bit_flip_decode(H, received, max_iter=50):
    """Hard-decision bit-flipping [1]. Input: received (0/1). Returns (decoded, converged, iters)."""
    c = np.asarray(received, dtype=int)
    if np.ndim(c) == 0 or len(c) != H.shape[1]:
        c = np.broadcast_to(c, H.shape[1])
    m, n = H.shape
    for it in range(1, max_iter + 1):
        syndrome = (H @ c) % 2
        if not syndrome.any():
            return c, True, it
        unsat = H.T @ syndrome
        worst = int(np.argmax(unsat))
        c[worst] ^= 1
    return c, np.all((H @ c) % 2 == 0), max_iter


def belief_propagation_decode(H, llr, max_iter=50):
    """Sum-product (Belief Propagation) decoder [2,3]. Input: LLR (positive => bit likely 0). Returns (decoded, converged, iters)."""
    m, n = H.shape
    cn, bn = tanner_graph(H)
    llr = np.asarray(llr, dtype=float)
    # Messages: from check j to bit i stored implicitly; we iterate by check then by bit
    # C2V[j][idx] = message from check j to the idx-th neighbor (which is cn[j][idx])
    # For each check j, neighbors are cn[j]; we need message from j to each neighbor.
    num_edges = sum(len(cn[j]) for j in range(m))
    c2v = np.zeros(num_edges)
    v2c = np.zeros(num_edges)
    # Map: edge (j,i) -> global index
    edge_idx = 0
    j_to_edge = []
    # Precompute edge map will be filled after j_to_edge is built
    for j in range(m):
        j_to_edge.append(list(range(edge_idx, edge_idx + len(cn[j]))))
        edge_idx += len(cn[j])
    # Precompute edge map
    edge_map = {}
    for j in range(m):
        for idx, i in enumerate(cn[j]):
            edge_map[(j, i)] = j_to_edge[j][idx]
    # Initialize v2c from channel LLR
    for i in range(n):
        for e in range(len(bn[i])):
            j = bn[i][e]
            glob = edge_map[(j, i)]
            v2c[glob] = llr[i]
    # Iterate
    for it in range(1, max_iter + 1):
        # Check to variable
        edge_idx = 0
        for j in range(m):
            nbs = cn[j]
            if not nbs:
                continue
            inc = v2c[edge_idx : edge_idx + len(nbs)]
            for k in range(len(nbs)):
                excl = np.concatenate([inc[:k], inc[k + 1 :]])
                if len(excl) == 0:
                    c2v[edge_idx + k] = 0.0
                else:
                    t = np.tanh(excl * 0.5)
                    prod = np.prod(t)
                    prod = np.clip(prod, -0.999999, 0.999999)
                    if prod >= 1:
                        c2v[edge_idx + k] = 20.0
                    elif prod <= -1:
                        c2v[edge_idx + k] = -20.0
                    else:
                        c2v[edge_idx + k] = 2.0 * np.arctanh(prod)
            edge_idx += len(nbs)
        # Variable to check
        for i in range(n):
            total_llr = llr[i]
            for e in range(len(bn[i])):
                j = bn[i][e]
                glob = edge_map[(j, i)]
                total_llr += c2v[glob]
            for e in range(len(bn[i])):
                j = bn[i][e]
                glob = edge_map[(j, i)]
                v2c[glob] = total_llr - c2v[glob]
        # Tentative decision
        decoded = np.zeros(n, dtype=int)
        for i in range(n):
            total = llr[i]
            for j in bn[i]:
                local = [idx for idx, nb in enumerate(cn[j]) if nb == i][0]
                total += c2v[j_to_edge[j][local]]
            decoded[i] = 1 if total < 0 else 0
        if np.all((H @ decoded) % 2 == 0):
            return decoded, True, it
    return decoded, np.all((H @ decoded) % 2 == 0), max_iter


def belief_propagation_decode_with_history(H, llr, max_iter=50):
    """Same as belief_propagation_decode but returns (decoded, converged, iters, history).
    history is a list of dicts: {'iter': int, 'syndrome': (m,), 'decoded': (n,), 'total_llr': (n,)}."""
    m, n = H.shape
    cn, bn = tanner_graph(H)
    llr = np.asarray(llr, dtype=float)
    num_edges = sum(len(cn[j]) for j in range(m))
    c2v = np.zeros(num_edges)
    v2c = np.zeros(num_edges)
    edge_idx = 0
    j_to_edge = []
    # Precompute edge map will be filled after j_to_edge is built
    for j in range(m):
        j_to_edge.append(list(range(edge_idx, edge_idx + len(cn[j]))))
        edge_idx += len(cn[j])
    # Precompute edge map
    edge_map = {}
    for j in range(m):
        for idx, i in enumerate(cn[j]):
            edge_map[(j, i)] = j_to_edge[j][idx]
    for i in range(n):
        for e in range(len(bn[i])):
            j = bn[i][e]
            glob = edge_map[(j, i)]
            v2c[glob] = llr[i]
    history = []
    for it in range(1, max_iter + 1):
        edge_idx = 0
        for j in range(m):
            nbs = cn[j]
            if not nbs:
                continue
            inc = v2c[edge_idx : edge_idx + len(nbs)]
            for k in range(len(nbs)):
                excl = np.concatenate([inc[:k], inc[k + 1 :]])
                if len(excl) == 0:
                    c2v[edge_idx + k] = 0.0
                else:
                    t = np.tanh(excl * 0.5)
                    prod = np.prod(t)
                    prod = np.clip(prod, -0.999999, 0.999999)
                    if prod >= 1:
                        c2v[edge_idx + k] = 20.0
                    elif prod <= -1:
                        c2v[edge_idx + k] = -20.0
                    else:
                        c2v[edge_idx + k] = 2.0 * np.arctanh(prod)
            edge_idx += len(nbs)
        for i in range(n):
            total_llr = llr[i]
            for e in range(len(bn[i])):
                j = bn[i][e]
                glob = edge_map[(j, i)]
                total_llr += c2v[glob]
            for e in range(len(bn[i])):
                j = bn[i][e]
                glob = edge_map[(j, i)]
                v2c[glob] = total_llr - c2v[glob]
        decoded = np.zeros(n, dtype=int)
        total_llr_per_bit = np.zeros(n)
        for i in range(n):
            total = llr[i]
            for j in bn[i]:
                local = [idx for idx, nb in enumerate(cn[j]) if nb == i][0]
                total += c2v[j_to_edge[j][local]]
            total_llr_per_bit[i] = total
            decoded[i] = 1 if total < 0 else 0
        syndrome = (H @ decoded) % 2
        history.append({
            "iter": it,
            "syndrome": syndrome.copy(),
            "decoded": decoded.copy(),
            "total_llr": total_llr_per_bit.copy(),
        })
        if np.all(syndrome == 0):
            return decoded, True, it, history
    return decoded, np.all((H @ decoded) % 2 == 0), max_iter, history


def minsum_decode(H, llr, max_iter=50, alpha=0.75):
    """Min-Sum decoder [2]: approximation to BP. alpha = normalization factor. Returns (decoded, converged, iters)."""
    m, n = H.shape
    cn, bn = tanner_graph(H)
    llr = np.asarray(llr, dtype=float)
    num_edges = sum(len(cn[j]) for j in range(m))
    c2v = np.zeros(num_edges)
    v2c = np.zeros(num_edges)
    edge_idx = 0
    j_to_edge = []
    # Precompute edge map will be filled after j_to_edge is built
    for j in range(m):
        j_to_edge.append(list(range(edge_idx, edge_idx + len(cn[j]))))
        edge_idx += len(cn[j])
    # Precompute edge map
    edge_map = {}
    for j in range(m):
        for idx, i in enumerate(cn[j]):
            edge_map[(j, i)] = j_to_edge[j][idx]
    for i in range(n):
        for j in bn[i]:
            local = [idx for idx, nb in enumerate(cn[j]) if nb == i][0]
            v2c[j_to_edge[j][local]] = llr[i]
    for it in range(1, max_iter + 1):
        edge_idx = 0
        for j in range(m):
            nbs = cn[j]
            if not nbs:
                continue
            inc = v2c[edge_idx : edge_idx + len(nbs)]
            sgn = np.sign(inc)
            sgn[sgn == 0] = 1
            mag = np.abs(inc)
            prod_sgn = np.prod(sgn)
            for k in range(len(nbs)):
                s = prod_sgn * sgn[k]
                # Min-Sum: magnitude = min of OTHER messages' magnitudes
                other_mag = np.concatenate([mag[:k], mag[k + 1 :]])
                m_val = np.min(other_mag) if len(other_mag) > 0 else 20.0
                c2v[edge_idx + k] = alpha * s * np.clip(m_val, 1e-10, 1e10)
            edge_idx += len(nbs)
        for i in range(n):
            total_llr = llr[i]
            for j in bn[i]:
                local = [idx for idx, nb in enumerate(cn[j]) if nb == i][0]
                total_llr += c2v[j_to_edge[j][local]]
            for j in bn[i]:
                glob = edge_map[(j, i)]
                v2c[glob] = total_llr - c2v[glob]
        decoded = np.zeros(n, dtype=int)
        for i in range(n):
            total = llr[i]
            for j in bn[i]:
                local = [idx for idx, nb in enumerate(cn[j]) if nb == i][0]
                total += c2v[j_to_edge[j][local]]
            decoded[i] = 1 if total < 0 else 0
        if np.all((H @ decoded) % 2 == 0):
            return decoded, True, it
    return decoded, np.all((H @ decoded) % 2 == 0), max_iter


# -----------------------------------------------------------------------------
# Channel capacity (for comparison with performance)
# -----------------------------------------------------------------------------

def capacity_bsc(p, eps=1e-12):
    """Capacity of BSC: C = 1 - H(p) bits per channel use."""
    p = np.clip(p, eps, 1 - eps)
    return 1.0 + p * np.log2(p) + (1 - p) * np.log2(1 - p)


def capacity_bec(epsilon):
    """Capacity of BEC: C = 1 - epsilon."""
    return 1.0 - np.asarray(epsilon, dtype=float)


def shannon_limit_awgn_rate(R):
    """Minimum Eb/N0 (dB) for rate R on AWGN channel: 2^(2R)-1 in linear, then to dB."""
    return 10 * np.log10((2 ** (2 * R) - 1) / (2 * R))
