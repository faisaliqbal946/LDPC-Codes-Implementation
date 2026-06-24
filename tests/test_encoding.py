import numpy as np

from ldpc import encode, is_codeword, syndrome


def test_random_messages_encode_to_valid_codewords(ldpc_code):
    H = ldpc_code["H"]
    G = ldpc_code["G"]
    col_order = ldpc_code["col_order"]
    K = ldpc_code["K"]
    N = ldpc_code["N"]
    rng = np.random.default_rng(7)

    for _ in range(20):
        message = rng.integers(0, 2, size=K)
        codeword = encode(G, message, col_order)

        assert codeword.shape == (N,)
        assert is_codeword(H, codeword)
        assert np.array_equal(syndrome(H, codeword), np.zeros(H.shape[0], dtype=int))
