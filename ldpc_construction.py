# ================== ENHANCED CONSTRUCTION ==================
# Improvements:
# - Added academic clarity on rate vs rank
# - Improved documentation for generator matrix
# - Clarified Gallager construction assumptions
# ===========================================================

"""
LDPC parity-check and generator matrix construction (Gallager regular).

Builds H for arbitrary block length n with target rate R = 1/2 and column weight wc = 3.
Row weight wr = 2*wc so that m = n*wc/wr = n/2 (regular rate-1/2 Gallager).

Requires: n divisible by wr (default wr = 6). If you need n ≈ 200, use 198 or 204.

Also provides: full_rank_H (GF(2) row reduction), systematic G, encode.
"""

import numpy as np


def build_H_gallager(n, wc, wr, seed=42):
    """Gallager regular LDPC: H is (m x n) with m = n*wc/wr, column weight wc, row weight wr."""
    rng = np.random.default_rng(seed)
    if n % wr != 0:
        raise ValueError(
            f"n={n} must be divisible by wr={wr} for this Gallager construction. "
            f"Try n={n - n % wr} or n={n + (wr - n % wr) % wr}."
        )
    if (n * wc) % wr != 0:
        raise ValueError(f"Invalid parameters: n*wc must be divisible by wr (got n={n}, wc={wc}, wr={wr})")

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


def full_rank_H(H):
    """Reduce H to full row rank over GF(2).
This ensures independent parity-check equations.
The resulting number of rows m = rank(H), so the actual code rate becomes R = (n - rank(H)) / n."""
    rref, pivots = gf2_rref(H)
    rank = len(pivots)
    return rref[:rank].copy()


def analyze_H(H):
    """Return structural diagnostics for a binary parity-check matrix.

    The 4-cycle count is computed from column-pair overlaps. If two columns
    share s parity checks, they contribute C(s, 2) distinct 4-cycles.
    """
    A = np.asarray(H, dtype=int) % 2
    if A.ndim != 2:
        raise ValueError("H must be a 2D parity-check matrix.")

    m, n = A.shape
    _, pivots = gf2_rref(A)
    rank = len(pivots)
    k = n - rank
    row_weights = A.sum(axis=1)
    col_weights = A.sum(axis=0)

    diagnostics = {
        "M": int(m),
        "N": int(n),
        "rank_gf2": int(rank),
        "K": int(k),
        "rate": float(k / n) if n else 0.0,
        "density": float(A.mean()) if A.size else 0.0,
        "row_weight_min": int(row_weights.min()) if m else 0,
        "row_weight_max": int(row_weights.max()) if m else 0,
        "row_weight_mean": float(row_weights.mean()) if m else 0.0,
        "column_weight_min": int(col_weights.min()) if n else 0,
        "column_weight_max": int(col_weights.max()) if n else 0,
        "column_weight_mean": float(col_weights.mean()) if n else 0.0,
        "estimated_4_cycles": None,
        "four_cycle_method": "column_pair_overlap",
        "four_cycle_feasible": False,
    }

    # For the project sizes this is exact and inexpensive. Keep a guard for
    # very large future matrices so analysis never dominates simulation setup.
    if n <= 5000:
        overlaps = A.T @ A
        upper = np.triu_indices(n, k=1)
        shared_checks = overlaps[upper]
        diagnostics["estimated_4_cycles"] = int(np.sum(shared_checks * (shared_checks - 1) // 2))
        diagnostics["four_cycle_feasible"] = True

    return diagnostics


def make_generator(H):
    """Systematic generator matrix construction.
After column permutation, H is in form [P | I_m].
Generator matrix is G = [I_k | P^T].
Returns G, column order, and k."""
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


def encode(G, message, col_order):
    """Encode k-bit message to n-bit codeword in original column order of H."""
    n = G.shape[1]
    c_perm = (np.asarray(message, dtype=int) @ G) % 2
    codeword = np.zeros(n, dtype=int)
    for i, orig in enumerate(col_order):
        codeword[orig] = c_perm[i]
    return codeword


def build_ldpc_rate_half(n, wc=3, seed=42, enforce_full_rank=True):
    """
    Build a regular Gallager LDPC code with design rate 1/2 and column weight wc.

    Uses wr = 2*wc so m_design = n/2. After full_rank_H, actual k = n - rank(H).

    Parameters
    ----------
    n : int
        Block length; must be divisible by 2*wc (e.g. for wc=3, n must be multiple of 6).
    wc : int
        Column weight (default 3).
    seed : int
        RNG seed for permutations.
    enforce_full_rank : bool
        If True (default), return H with independent rows only.

    Returns
    -------
    H : ndarray (m x n)
    G : ndarray (k x n)
    col_order : ndarray
    k, m : int
    rate : float
        Actual k / n after rank reduction.
    """
    wr = 2 * wc
    if n % wr != 0:
        raise ValueError(
            f"For rate 1/2 and wc={wc}, need wr={wr} and n divisible by {wr}. "
            f"Got n={n}. Examples: n=198, 204, 498, 504 for wc=3."
        )

    H_full = build_H_gallager(n, wc, wr, seed=seed)
    if enforce_full_rank:
        H = full_rank_H(H_full)
    else:
        H = H_full
    G, col_order, k = make_generator(H)
    m = H.shape[0]
    rate = k / n
    return H, G, col_order, k, m, rate


def nearest_valid_n(n_target, wc=3):
    """Largest n <= n_target divisible by 2*wc (for rate-1/2 Gallager)."""
    wr = 2 * wc
    return n_target - (n_target % wr)


def smallest_valid_n_ge(n_target, wc=3):
    """Smallest n >= n_target divisible by 2*wc."""
    wr = 2 * wc
    rem = n_target % wr
    if rem == 0:
        return n_target
    return n_target + (wr - rem)


def build_ldpc_for_target_length(n_target, wc=3, seed=42, prefer="ge"):
    """
    Build rate-1/2 Gallager LDPC with column weight wc for a target block length.

    Gallager's row blocks require n divisible by wr = 2*wc (e.g. 6 when wc=3).
    n=200 and n=500 are not multiples of 6, so we use the nearest valid length:
      - n_target=200 -> n=204 (prefer='ge') or n=198 (prefer='le')
      - n_target=500 -> n=504 (prefer='ge') or n=498 (prefer='le')

    Returns the same tuple as build_ldpc_rate_half plus the resolved block length n.
    """
    if prefer == "le":
        n = nearest_valid_n(n_target, wc=wc)
    else:
        n = smallest_valid_n_ge(n_target, wc=wc)
    H, G, col_order, k, m, rate = build_ldpc_rate_half(n, wc=wc, seed=seed)
    return H, G, col_order, k, m, rate, n
