import numpy as np
import pytest

from ldpc_construction import build_ldpc_for_target_length


@pytest.fixture(scope="session")
def ldpc_code():
    H, G, col_order, K, M, rate, N = build_ldpc_for_target_length(
        200, wc=3, seed=42, prefer="ge"
    )
    return {
        "H": H,
        "G": G,
        "col_order": col_order,
        "K": K,
        "M": M,
        "rate": rate,
        "N": N,
        "rng": np.random.default_rng(1234),
    }
