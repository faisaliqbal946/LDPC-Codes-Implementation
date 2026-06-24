import numpy as np

from ldpc import awgn_channel, bec_channel, bsc


def test_bsc_output_shape_and_approximate_flip_rate():
    codeword = np.zeros(10_000, dtype=int)
    p = 0.08

    received, error_mask = bsc(codeword, p, seed=11)

    assert received.shape == codeword.shape
    assert error_mask.shape == codeword.shape
    assert np.isclose(error_mask.mean(), p, atol=0.015)
    assert np.array_equal(received, error_mask)


def test_bec_output_has_erasures_and_same_shape():
    codeword = np.zeros(2_000, dtype=int)

    received = bec_channel(codeword, 0.2, seed=12)

    assert received.shape == codeword.shape
    assert np.any(received == -1.0)
    assert set(np.unique(received)).issubset({-1.0, 0.0})


def test_awgn_output_shape_and_positive_sigma2():
    codeword = np.zeros(512, dtype=int)

    received, sigma2 = awgn_channel(codeword, snr_db=3.0, rate=0.5, seed=13)

    assert received.shape == codeword.shape
    assert sigma2 > 0
    assert np.isfinite(received).all()
