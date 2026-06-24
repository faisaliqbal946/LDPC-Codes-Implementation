# NOTE:
# This file uses a small (20,10) LDPC code for demonstration and visualization.
# Due to the short block length, performance is limited and does not reflect
# practical LDPC performance. For realistic results, see experiment.py.

#!/usr/bin/env python3
"""
Experiments: decoder comparison and channel comparison with capacity bounds.

For large block length (n ~ 200) and 1000-frame sweeps, use experiment.py instead.

Generates:
  - BER vs BSC flip probability (Bit-Flip, BP, Min-Sum) and BSC capacity
  - BER vs AWGN Eb/N0 (BP, Min-Sum) and Shannon limit
  - BER vs BEC erasure probability (BP, Min-Sum) and BEC capacity
  - Decoder comparison on one channel

References:
  [1] Gallager, "Low-Density Parity-Check Codes," 1963.
  [2] MacKay, "Information Theory, Inference, and Learning Algorithms," Ch. 47.
  [3] Richardson & Urbanke, IEEE Trans. IT, 2001.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ldpc import (
    build_H,
    make_generator,
    full_rank_H,
    encode,
    bsc,
    bsc_llr,
    awgn_channel,
    awgn_llr,
    bec_channel,
    bec_llr,
    bit_flip_decode,
    belief_propagation_decode,
    minsum_decode,
    capacity_bsc,
    capacity_bec,
    shannon_limit_awgn_rate,
)

# (20,10) code: 10x20 full-rank H, syndrome length = 10
N, WC, WR = 20, 3, 5
SEED = 0
H_full = build_H(N, WC, WR, seed=SEED)
H = full_rank_H(H_full)  # 10 x 20, rank 10
G, col_order, K = make_generator(H)
RATE = K / N
M = H.shape[0]
assert M == 10 and K == 10, "Expected (20,10) code with 10-bit syndrome"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RNG = np.random.default_rng(42)


def run_ber_bsc(flip_probs, frames_per_point=800):
    """BER vs BSC p for Bit-Flip, BP, Min-Sum. Return dict decoder_name -> BER array."""
    results = {"Bit-Flip": [], "BP": [], "Min-Sum": []}
    for p in flip_probs:
        errs = {"Bit-Flip": 0, "BP": 0, "Min-Sum": 0}
        bits = 0
        for _ in range(frames_per_point):
            msg = RNG.integers(0, 2, size=K)
            cw = encode(G, msg, col_order)
            rx, _ = bsc(cw, p, seed=RNG.integers(0, 2**31))
            llr = bsc_llr(rx, p)
            dec_bf, _, _ = bit_flip_decode(H, rx)
            dec_bp, _, _ = belief_propagation_decode(H, llr)
            dec_ms, _, _ = minsum_decode(H, llr)
            errs["Bit-Flip"] += np.sum(dec_bf != cw)
            errs["BP"] += np.sum(dec_bp != cw)
            errs["Min-Sum"] += np.sum(dec_bp != cw)
            bits += N
        for name in results:
            results[name].append(errs[name] / bits)
    return results


def run_ber_awgn(snr_db_list, frames_per_point=600):
    """BER vs Eb/N0 for BP and Min-Sum."""
    results = {"BP": [], "Min-Sum": []}
    for snr_db in snr_db_list:
        errs = {"BP": 0, "Min-Sum": 0}
        bits = 0
        for _ in range(frames_per_point):
            msg = RNG.integers(0, 2, size=K)
            cw = encode(G, msg, col_order)
            rx, sigma2 = awgn_channel(cw, snr_db, rate=RATE, seed=RNG.integers(0, 2**31))
            llr = awgn_llr(rx, sigma2)
            dec_bp, _, _ = belief_propagation_decode(H, llr)
            dec_ms, _, _ = minsum_decode(H, llr)
            errs["BP"] += np.sum(dec_bp != cw)
            errs["Min-Sum"] += np.sum(dec_bp != cw)
            bits += N
        for name in results:
            results[name].append(errs[name] / bits)
    return results


def run_ber_bec(eps_list, frames_per_point=600):
    """BER vs BEC epsilon for BP and Min-Sum."""
    results = {"BP": [], "Min-Sum": []}
    for eps in eps_list:
        errs = {"BP": 0, "Min-Sum": 0}
        bits = 0
        for _ in range(frames_per_point):
            msg = RNG.integers(0, 2, size=K)
            cw = encode(G, msg, col_order)
            rx = bec_channel(cw, eps, seed=RNG.integers(0, 2**31))
            llr = bec_llr(rx)
            dec_bp, _, _ = belief_propagation_decode(H, llr)
            dec_ms, _, _ = minsum_decode(H, llr)
            errs["BP"] += np.sum(dec_bp != cw)
            errs["Min-Sum"] += np.sum(dec_bp != cw)
            bits += N
        for name in results:
            results[name].append(errs[name] / bits)
    return results


def fig_bsc_with_capacity():
    """BER vs BSC p; plot capacity C(p) = 1 - H(p) as reference."""
    flip_probs = np.linspace(0.02, 0.18, 10)
    decoders = {
        "Bit-Flip": None,
        "BP": None,
        "Min-Sum": None,
    }
    res = run_ber_bsc(flip_probs, frames_per_point=300)
    cap = [capacity_bsc(p) for p in flip_probs]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.semilogy(flip_probs, res["Bit-Flip"], "-o", label="Bit-Flip (hard)", markersize=5)
    ax1.semilogy(flip_probs, res["BP"], "-s", label="BP (soft)", markersize=5)
    ax1.semilogy(flip_probs, res["Min-Sum"], "-^", label="Min-Sum (soft)", markersize=5)
    ax1.semilogy(flip_probs, flip_probs, "k--", lw=1, label="Uncoded (BER = p)")
    ax1.set_xlabel("BSC Crossover Probability p")
    ax1.set_ylabel("Bit Error Rate (BER)")
    ax1.set_ylim(bottom=1e-5)
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("(20,10) LDPC on BSC: BER vs p (decoder comparison)")

    ax2 = ax1.twinx()
    ax2.plot(flip_probs, cap, "k:", lw=1.5, label="Capacity C(p) = 1 - H(p)")
    ax2.set_ylabel("Capacity (bits/channel use)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "ber_bsc_with_capacity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)
    return path



from math import erfc

def uncoded_awgn(snr_db):
    gamma = 10 ** (snr_db / 10)
    return 0.5 * erfc(np.sqrt(gamma))

def fig_awgn_with_shannon():
    """BER vs Eb/N0 for AWGN; show Shannon limit for rate R."""
    snr_db = np.linspace(0, 6, 10)
    res = run_ber_awgn(snr_db, frames_per_point=250)
    shannon_db = shannon_limit_awgn_rate(RATE)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(snr_db, res["BP"], "-s", label="BP", markersize=5)
    unc = [uncoded_awgn(x) for x in snr_db]
    ax.semilogy(snr_db, unc, "k--", label="Uncoded BPSK")
    ax.semilogy(snr_db, res["Min-Sum"], "-^", label="Min-Sum", markersize=5)
    ax.axvline(shannon_db, color="k", ls="--", lw=1, label=f"Shannon limit R={RATE:.2f} (~{shannon_db:.2f} dB)")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_ylim(bottom=1e-5)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("(20,10) LDPC (demonstration): BER vs Eb/N0 (AWGN)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "ber_awgn_with_shannon.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)
    return path


def fig_bec_with_capacity():
    """BER vs BEC epsilon; show capacity C(eps) = 1 - eps."""
    eps_list = np.linspace(0.05, 0.5, 10)
    res = run_ber_bec(eps_list, frames_per_point=250)
    cap = capacity_bec(eps_list)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.semilogy(eps_list, res["BP"], "-s", label="BP", markersize=5)
    ax1.semilogy(eps_list, res["Min-Sum"], "-^", label="Min-Sum", markersize=5)
    ax1.semilogy(eps_list, eps_list, "k--", lw=1, label="Uncoded (BER = eps)")
    ax1.set_xlabel("BEC Erasure Probability epsilon")
    ax1.set_ylabel("Bit Error Rate (BER)")
    ax1.set_ylim(bottom=1e-5)
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("(20,10) LDPC on BEC: BER vs epsilon")

    ax2 = ax1.twinx()
    ax2.plot(eps_list, cap, "k:", lw=1.5, label="Capacity C = 1 - epsilon")
    ax2.set_ylabel("Capacity (bits/channel use)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "ber_bec_with_capacity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)
    return path


def fig_channel_comparison():
    """One figure: BER curves for BSC, AWGN, BEC (BP only) with capacity / limit."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # BSC
    fp = np.linspace(0.02, 0.18, 8)
    res_bsc = run_ber_bsc(fp, frames_per_point=200)
    cap_bsc = [capacity_bsc(p) for p in fp]
    ax = axes[0]
    ax.semilogy(fp, res_bsc["BP"], "-o", label="LDPC BP", markersize=4)
    ax.semilogy(fp, fp, "k--", lw=1, label="Uncoded")
    ax.set_xlabel("BSC p")
    ax.set_ylabel("BER")
    ax.set_title("BSC")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(fp, cap_bsc, "k:", lw=1, label="C(p)")
    ax2.set_ylabel("Capacity", fontsize=8, color="gray")
    ax2.set_ylim(0, 1)

    # AWGN
    snr = np.linspace(0.5, 5.5, 8)
    res_awgn = run_ber_awgn(snr, frames_per_point=200)
    ax = axes[1]
    ax.semilogy(snr, res_awgn["BP"], "-o", label="LDPC BP", markersize=4)
    ax.axvline(shannon_limit_awgn_rate(RATE), color="k", ls="--", lw=1, label="Shannon limit")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title("AWGN")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # BEC
    eps = np.linspace(0.05, 0.5, 8)
    res_bec = run_ber_bec(eps, frames_per_point=200)
    cap_bec = capacity_bec(eps)
    ax = axes[2]
    ax.semilogy(eps, res_bec["BP"], "-o", label="LDPC BP", markersize=4)
    ax.semilogy(eps, eps, "k--", lw=1, label="Uncoded")
    ax.set_xlabel("BEC epsilon")
    ax.set_ylabel("BER")
    ax.set_title("BEC")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(eps, cap_bec, "k:", lw=1, label="C(eps)")
    ax2.set_ylabel("Capacity", fontsize=8, color="gray")
    ax2.set_ylim(0, 1)

    fig.suptitle("(20,10) LDPC: Channel comparison (BER vs capacity / Shannon limit)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "channel_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)
    return path


def main():
    print("(20,10) LDPC, syndrome length m =", M)
    print("Running experiments...")
    fig_bsc_with_capacity()
    fig_awgn_with_shannon()
    fig_bec_with_capacity()
    fig_channel_comparison()
    print("Done. Results in", RESULTS_DIR)


if __name__ == "__main__":
    main()
