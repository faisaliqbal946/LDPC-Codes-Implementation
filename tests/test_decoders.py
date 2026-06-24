import numpy as np

from ldpc import (
    awgn_channel,
    awgn_llr,
    belief_propagation_decode,
    bit_flip_decode,
    bsc,
    bsc_llr,
    encode,
    is_codeword,
    minsum_decode,
)


def _assert_decoder_result(result, N):
    decoded, converged, iterations = result
    assert decoded.shape == (N,)
    assert isinstance(converged, (bool, np.bool_))
    assert isinstance(iterations, (int, np.integer))
    assert iterations >= 1


def test_decoders_return_decoded_converged_and_iterations(ldpc_code):
    H = ldpc_code["H"]
    G = ldpc_code["G"]
    col_order = ldpc_code["col_order"]
    K = ldpc_code["K"]
    N = ldpc_code["N"]
    rng = np.random.default_rng(21)
    message = rng.integers(0, 2, size=K)
    codeword = encode(G, message, col_order)

    bsc_received, _ = bsc(codeword, p=0.01, seed=22)
    bsc_llrs = bsc_llr(bsc_received, p=0.01)
    awgn_received, sigma2 = awgn_channel(codeword, snr_db=8.0, rate=ldpc_code["rate"], seed=23)
    awgn_llrs = awgn_llr(awgn_received, sigma2)

    _assert_decoder_result(bit_flip_decode(H, bsc_received, max_iter=15), N)
    _assert_decoder_result(belief_propagation_decode(H, awgn_llrs, max_iter=10), N)
    _assert_decoder_result(minsum_decode(H, bsc_llrs, max_iter=10), N)


def test_bp_recovers_original_codeword_at_very_low_noise(ldpc_code):
    H = ldpc_code["H"]
    G = ldpc_code["G"]
    col_order = ldpc_code["col_order"]
    K = ldpc_code["K"]
    rng = np.random.default_rng(24)
    message = rng.integers(0, 2, size=K)
    codeword = encode(G, message, col_order)

    received, sigma2 = awgn_channel(codeword, snr_db=12.0, rate=ldpc_code["rate"], seed=25)
    llr = awgn_llr(received, sigma2)
    decoded, converged, iterations = belief_propagation_decode(H, llr, max_iter=20)

    assert converged
    assert iterations >= 1
    assert is_codeword(H, decoded)
    assert np.array_equal(decoded, codeword)
