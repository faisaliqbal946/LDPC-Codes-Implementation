# ================== ENHANCED SETTINGS APPLIED ==================
# FRAMES_PER_POINT increased to 300 for stable BER estimation
# MAX_ITER_BP_MS increased to 50 for better BP convergence
# AWGN validation relaxed (not required at all SNR points)
# ===============================================================

#!/usr/bin/env python3
"""
Large-block LDPC Monte Carlo experiments (n ~ 200 class).

Uses ldpc_construction (Gallager regular, wc=3, design rate 1/2) and ldpc channels/decoders.
Target CODE_LENGTH=200 -> actual n=204 (smallest n>=200 divisible by wr=6).

Why default settings are modest (not "max everything"):
  - Decoders are pure Python over the Tanner graph: each frame runs BP / Min-Sum for many
    iterations. For n ~ 200, that is CPU-bound, not memory-bound (H is small in MB terms).
  - Raising frames (e.g. 1000) and sweep points multiplies runtime roughly linearly; on a
    laptop this can mean tens of minutes to many hours for one full run.
  - Very large n (500+) or huge frame counts are what we avoid for interactive / coursework
    machines: not because H "does not fit", but because Monte Carlo + iterative decoding
    becomes too slow to finish reliably.

Run:
  python experiment.py
  EXPERIMENT_FRAMES=50 python experiment.py   # smoke test
  EXPERIMENT_FRAMES=250 python experiment.py  # default-ish (edit defaults below)
  EXPERIMENT_FRAMES=1000 python experiment.py # smoother curves (slow)

Outputs:
  results/figures/
  bsc_ber_fer.png
  awgn_ber_fer.png
  bec_ber_fer.png
  decoder_convergence.png
  coding_gain_summary.png
  results/tables/
  bsc_results.csv
  awgn_results.csv
  bec_results.csv
  results/summaries/
  experiment_summary.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import csv
import sys
import time
from math import erfc

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ldpc import (
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
    encode,
)
from ldpc_construction import build_ldpc_for_target_length

# --- User-requested parameters -------------------------------------------------
CODE_LENGTH_TARGET = 200  # Gallager wr=6 => use n=204 (see build_ldpc_for_target_length)
# Defaults are minimal so a laptop finishes in minutes. Override env vars for heavier runs.
FRAMES_PER_POINT = int(os.environ.get("EXPERIMENT_FRAMES", "300"))
MAX_ITER_BP_MS = int(os.environ.get("EXPERIMENT_MAX_ITER", "50"))
MAX_ITER_BF = int(os.environ.get("EXPERIMENT_MAX_ITER_BF", "15"))
# Gallager permutations; try another seed if validation warns (see EXPERIMENT_CONSTRUCTION_SEED).
CONSTRUCTION_SEED = int(os.environ.get("EXPERIMENT_CONSTRUCTION_SEED", "42"))
RNG_SEED = 12345

BSC_P = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18])
AWGN_EBNO_DB = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
BEC_EPS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])

# EXPERIMENT_QUICK=1: fewer sweep points (still shows shape of curves). Use for very slow machines.
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


if _env_bool("EXPERIMENT_QUICK", default=False):
    BSC_P = BSC_P[::3]  # 6 points
    AWGN_EBNO_DB = AWGN_EBNO_DB[::2]  # 4 points
    BEC_EPS = BEC_EPS[::2]  # 3 points

# Assignment reference line (rate-1/2 BPSK / capacity literature); plotted alongside ldpc rate.
SHANNON_REF_DB = 0.187

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
SUMMARIES_DIR = os.path.join(RESULTS_DIR, "summaries")
for _dir in (FIGURES_DIR, TABLES_DIR, SUMMARIES_DIR):
    os.makedirs(_dir, exist_ok=True)
RNG = np.random.default_rng(RNG_SEED)

H, G, col_order, K, M, RATE, N = build_ldpc_for_target_length(
    CODE_LENGTH_TARGET, wc=3, seed=CONSTRUCTION_SEED, prefer="ge"
)


def _empty_counts(decoders: list[str]) -> dict:
    return {
        decoder: {"bit_errors": 0, "frame_errors": 0, "converged": 0, "iterations": 0, "frames": 0, "bits": 0}
        for decoder in decoders
    }


def _record_decode(counts: dict, decoder: str, decoded: np.ndarray, converged: bool, iters: int, codeword: np.ndarray) -> None:
    bit_errors = int(np.sum(decoded != codeword))
    counts[decoder]["bit_errors"] += bit_errors
    counts[decoder]["frame_errors"] += int(bit_errors > 0)
    counts[decoder]["converged"] += int(bool(converged))
    counts[decoder]["iterations"] += int(iters)
    counts[decoder]["frames"] += 1
    counts[decoder]["bits"] += int(len(codeword))


def _counts_to_metrics(counts: dict) -> dict:
    metrics = {}
    for decoder, c in counts.items():
        frames = max(int(c["frames"]), 1)
        bits = max(int(c["bits"]), 1)
        metrics[decoder] = {
            "ber": float(c["bit_errors"] / bits),
            "fer": float(c["frame_errors"] / frames),
            "convergence_rate": float(c["converged"] / frames),
            "avg_iterations": float(c["iterations"] / frames),
        }
    return metrics


def _merge_point_metrics(rows: list[dict], decoders: list[str]) -> dict:
    return {
        decoder: {
            metric: np.array([row[decoder][metric] for row in rows], dtype=float)
            for metric in ("ber", "fer", "convergence_rate", "avg_iterations")
        }
        for decoder in decoders
    }


def _coding_gain_db(uncoded_ber: float, coded_ber: float) -> float:
    if coded_ber <= 0:
        return float("inf")
    if uncoded_ber <= 0:
        return float("-inf")
    return float(10.0 * np.log10(float(uncoded_ber) / float(coded_ber)))


def _pool_workers() -> int:
    raw = os.environ.get("EXPERIMENT_WORKERS", "").strip()
    if raw:
        return max(1, int(raw))
    # Default: single process (lowest CPU load). Set EXPERIMENT_WORKERS=4 to parallelize sweeps.
    return 1


def _bsc_point_task(args):
    """Worker: Monte Carlo at one BSC p (picklable for multiprocessing)."""
    H, G, col_order, K, N, p, frames, base_seed, max_iter_bp, max_iter_bf = args
    rng = np.random.default_rng(base_seed)
    from ldpc import (
        encode,
        bsc,
        bsc_llr,
        bit_flip_decode,
        belief_propagation_decode,
        minsum_decode,
    )

    counts = _empty_counts(["Bit-Flip", "BP", "Min-Sum"])
    for _ in range(frames):
        msg = rng.integers(0, 2, size=K)
        cw = encode(G, msg, col_order)
        rx, _ = bsc(cw, float(p), seed=rng.integers(0, 2**31))
        llr = bsc_llr(rx, float(p))
        d_bf, c_bf, i_bf = bit_flip_decode(H, rx, max_iter=max_iter_bf)
        d_bp, c_bp, i_bp = belief_propagation_decode(H, llr, max_iter=max_iter_bp)
        d_ms, c_ms, i_ms = minsum_decode(H, llr, max_iter=max_iter_bp)
        _record_decode(counts, "Bit-Flip", d_bf, c_bf, i_bf, cw)
        _record_decode(counts, "BP", d_bp, c_bp, i_bp, cw)
        _record_decode(counts, "Min-Sum", d_ms, c_ms, i_ms, cw)
    return _counts_to_metrics(counts)


def _awgn_point_task(args):
    H, G, col_order, K, N, rate, snr_db, frames, base_seed, max_iter_bp = args
    rng = np.random.default_rng(base_seed)
    from ldpc import encode, awgn_channel, awgn_llr, belief_propagation_decode, minsum_decode

    counts = _empty_counts(["BP", "Min-Sum"])
    for _ in range(frames):
        msg = rng.integers(0, 2, size=K)
        cw = encode(G, msg, col_order)
        rx, sigma2 = awgn_channel(cw, float(snr_db), rate=rate, seed=rng.integers(0, 2**31))
        llr = awgn_llr(rx, sigma2)
        d_bp, c_bp, i_bp = belief_propagation_decode(H, llr, max_iter=max_iter_bp)
        d_ms, c_ms, i_ms = minsum_decode(H, llr, max_iter=max_iter_bp)
        _record_decode(counts, "BP", d_bp, c_bp, i_bp, cw)
        _record_decode(counts, "Min-Sum", d_ms, c_ms, i_ms, cw)
    return _counts_to_metrics(counts)


def _bec_point_task(args):
    H, G, col_order, K, N, eps, frames, base_seed, max_iter_bp = args
    rng = np.random.default_rng(base_seed)
    from ldpc import encode, bec_channel, bec_llr, belief_propagation_decode, minsum_decode

    counts = _empty_counts(["BP", "Min-Sum"])
    for _ in range(frames):
        msg = rng.integers(0, 2, size=K)
        cw = encode(G, msg, col_order)
        rx = bec_channel(cw, float(eps), seed=rng.integers(0, 2**31))
        llr = bec_llr(rx)
        d_bp, c_bp, i_bp = belief_propagation_decode(H, llr, max_iter=max_iter_bp)
        d_ms, c_ms, i_ms = minsum_decode(H, llr, max_iter=max_iter_bp)
        _record_decode(counts, "BP", d_bp, c_bp, i_bp, cw)
        _record_decode(counts, "Min-Sum", d_ms, c_ms, i_ms, cw)
    return _counts_to_metrics(counts)


def _ber_uncoded_awgn_bpsk_db(snr_db: np.ndarray) -> np.ndarray:
    """Matches ldpc.awgn_channel scaling for uncoded BPSK (rate=1): Pe = 0.5*erfc(sqrt(Eb/N0))."""
    gamma = 10.0 ** (snr_db / 10.0)
    gamma = np.maximum(gamma, 1e-300)
    # numpy does not always ship erfc; keep it stdlib-only.
    return np.array([0.5 * erfc(float(np.sqrt(g))) for g in gamma], dtype=float)


def run_awgn_uncoded_empirical(snr_db_list: np.ndarray) -> np.ndarray:
    """Uncoded BPSK on K bits per frame, same Eb/N0 definition as ldpc.awgn_channel (rate=1)."""
    out: list[float] = []
    for snr_db in snr_db_list:
        errs = bits = 0
        for _ in range(FRAMES_PER_POINT):
            msg = RNG.integers(0, 2, size=K)
            rx, _sigma2 = awgn_channel(msg, float(snr_db), rate=1.0, seed=RNG.integers(0, 2**31))
            dec = (rx < 0.0).astype(int)
            errs += int(np.sum(dec != msg))
            bits += K
        out.append(errs / bits)
    return np.array(out, dtype=float)


def run_bsc(flip_probs: np.ndarray) -> dict:
    decoders = ["Bit-Flip", "BP", "Min-Sum"]
    w = _pool_workers()
    if w == 1:
        rows = []
        for p in flip_probs:
            counts = _empty_counts(decoders)
            for _ in range(FRAMES_PER_POINT):
                msg = RNG.integers(0, 2, size=K)
                cw = encode(G, msg, col_order)
                rx, _ = bsc(cw, float(p), seed=RNG.integers(0, 2**31))
                llr = bsc_llr(rx, float(p))
                d_bf, c_bf, i_bf = bit_flip_decode(H, rx, max_iter=MAX_ITER_BF)
                d_bp, c_bp, i_bp = belief_propagation_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                d_ms, c_ms, i_ms = minsum_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                _record_decode(counts, "Bit-Flip", d_bf, c_bf, i_bf, cw)
                _record_decode(counts, "BP", d_bp, c_bp, i_bp, cw)
                _record_decode(counts, "Min-Sum", d_ms, c_ms, i_ms, cw)
            rows.append(_counts_to_metrics(counts))
        return _merge_point_metrics(rows, decoders)

    args = []
    for i, p in enumerate(flip_probs):
        seed_i = int(RNG_SEED + 1_000_003 * i + int(round(float(p) * 1e6)))
        args.append(
            (
                H.copy(),
                G.copy(),
                col_order.copy(),
                K,
                N,
                float(p),
                FRAMES_PER_POINT,
                seed_i,
                MAX_ITER_BP_MS,
                MAX_ITER_BF,
            )
        )
    with mp.Pool(w) as pool:
        rows = pool.map(_bsc_point_task, args)
    return _merge_point_metrics(rows, decoders)


def run_awgn(snr_db_list: np.ndarray) -> dict:
    decoders = ["BP", "Min-Sum"]
    w = _pool_workers()
    if w == 1:
        rows = []
        for snr_db in snr_db_list:
            counts = _empty_counts(decoders)
            for _ in range(FRAMES_PER_POINT):
                msg = RNG.integers(0, 2, size=K)
                cw = encode(G, msg, col_order)
                rx, sigma2 = awgn_channel(cw, float(snr_db), rate=RATE, seed=RNG.integers(0, 2**31))
                llr = awgn_llr(rx, sigma2)
                d_bp, c_bp, i_bp = belief_propagation_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                d_ms, c_ms, i_ms = minsum_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                _record_decode(counts, "BP", d_bp, c_bp, i_bp, cw)
                _record_decode(counts, "Min-Sum", d_ms, c_ms, i_ms, cw)
            rows.append(_counts_to_metrics(counts))
        return _merge_point_metrics(rows, decoders)

    args = []
    for i, snr_db in enumerate(snr_db_list):
        seed_i = int(RNG_SEED + 2_000_009 * i + int(float(snr_db) * 1000))
        args.append(
            (
                H.copy(),
                G.copy(),
                col_order.copy(),
                K,
                N,
                float(RATE),
                float(snr_db),
                FRAMES_PER_POINT,
                seed_i,
                MAX_ITER_BP_MS,
            )
        )
    with mp.Pool(w) as pool:
        rows = pool.map(_awgn_point_task, args)
    return _merge_point_metrics(rows, decoders)


def run_bec(eps_list: np.ndarray) -> dict:
    decoders = ["BP", "Min-Sum"]
    w = _pool_workers()
    if w == 1:
        rows = []
        for eps in eps_list:
            counts = _empty_counts(decoders)
            for _ in range(FRAMES_PER_POINT):
                msg = RNG.integers(0, 2, size=K)
                cw = encode(G, msg, col_order)
                rx = bec_channel(cw, float(eps), seed=RNG.integers(0, 2**31))
                llr = bec_llr(rx)
                d_bp, c_bp, i_bp = belief_propagation_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                d_ms, c_ms, i_ms = minsum_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                _record_decode(counts, "BP", d_bp, c_bp, i_bp, cw)
                _record_decode(counts, "Min-Sum", d_ms, c_ms, i_ms, cw)
            rows.append(_counts_to_metrics(counts))
        return _merge_point_metrics(rows, decoders)

    args = []
    for i, eps in enumerate(eps_list):
        seed_i = int(RNG_SEED + 3_000_011 * i + int(round(float(eps) * 1e6)))
        args.append(
            (
                H.copy(),
                G.copy(),
                col_order.copy(),
                K,
                N,
                float(eps),
                FRAMES_PER_POINT,
                seed_i,
                MAX_ITER_BP_MS,
            )
        )
    with mp.Pool(w) as pool:
        rows = pool.map(_bec_point_task, args)
    return _merge_point_metrics(rows, decoders)


DECODER_STYLE = {
    "Bit-Flip": {"color": "#7f7f7f", "marker": "o", "linestyle": "-"},
    "BP": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    "Min-Sum": {"color": "#d62728", "marker": "^", "linestyle": "-"},
}


def _log_plot_values(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), floor)


def _uncoded_fer_from_ber(ber: np.ndarray, frame_len: int = N) -> np.ndarray:
    ber = np.clip(np.asarray(ber, dtype=float), 0.0, 1.0)
    return 1.0 - np.power(1.0 - ber, frame_len)


def _style_axis(ax) -> None:
    ax.grid(True, which="both", alpha=0.28, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=9)


def _plot_decoder_metric(ax, x: np.ndarray, res: dict, metric: str, decoders: list[str]) -> None:
    for decoder in decoders:
        style = DECODER_STYLE[decoder]
        ax.semilogy(
            x,
            _log_plot_values(res[decoder][metric]),
            label=decoder,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=4.5,
        )


def plot_bsc_ber_fer(res: dict, uncoded_ber: np.ndarray, path: str) -> None:
    cap = np.array([capacity_bsc(float(p)) for p in BSC_P])
    uncoded_fer = _uncoded_fer_from_ber(uncoded_ber)
    decoders = ["Bit-Flip", "BP", "Min-Sum"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    fig.suptitle(f"({N},{K}) LDPC on BSC: BER/FER vs p", fontsize=13, fontweight="bold")

    ax = axes[0]
    _plot_decoder_metric(ax, BSC_P, res, "ber", decoders)
    ax.semilogy(BSC_P, _log_plot_values(uncoded_ber), "k--", linewidth=1.7, label="Uncoded baseline")
    ax.set_xlabel("BSC crossover probability p")
    ax.set_ylabel("BER (log scale)")
    ax.set_title("Bit error rate")
    _style_axis(ax)

    ax = axes[1]
    _plot_decoder_metric(ax, BSC_P, res, "fer", decoders)
    ax.semilogy(BSC_P, _log_plot_values(uncoded_fer), "k--", linewidth=1.7, label="Uncoded baseline")
    ax.set_xlabel("BSC crossover probability p")
    ax.set_ylabel("FER (log scale)")
    ax.set_title("Frame error rate")
    _style_axis(ax)

    ax_ref = axes[1].twinx()
    ax_ref.plot(BSC_P, cap, color="#555555", linestyle=":", linewidth=1.4, label="Theoretical capacity")
    ax_ref.set_ylabel("Capacity reference (bits/use)", color="#555555")
    ax_ref.tick_params(axis="y", labelcolor="#555555", labelsize=9)
    ax_ref.set_ylim(0.0, 1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = ax_ref.get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_awgn_ber_fer(res: dict, unc_emp: np.ndarray, path: str) -> None:
    unc_an = _ber_uncoded_awgn_bpsk_db(AWGN_EBNO_DB)
    shannon_model_db = float(shannon_limit_awgn_rate(RATE))
    uncoded_fer = _uncoded_fer_from_ber(unc_emp)
    decoders = ["BP", "Min-Sum"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    fig.suptitle(f"({N},{K}) LDPC on AWGN: BER/FER vs Eb/N0", fontsize=13, fontweight="bold")

    ax = axes[0]
    _plot_decoder_metric(ax, AWGN_EBNO_DB, res, "ber", decoders)
    ax.semilogy(AWGN_EBNO_DB, _log_plot_values(unc_emp), "k--", linewidth=1.7, label="Uncoded baseline")
    ax.semilogy(AWGN_EBNO_DB, _log_plot_values(unc_an), color="#555555", linestyle=":", linewidth=1.2, label="Uncoded analytic")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER (log scale)")
    ax.set_title("Bit error rate")
    _style_axis(ax)

    ax = axes[1]
    _plot_decoder_metric(ax, AWGN_EBNO_DB, res, "fer", decoders)
    ax.semilogy(AWGN_EBNO_DB, _log_plot_values(uncoded_fer), "k--", linewidth=1.7, label="Uncoded baseline")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("FER (log scale)")
    ax.set_title("Frame error rate")
    _style_axis(ax)

    for ax in axes:
        ax.axvline(SHANNON_REF_DB, color="#777777", linestyle="--", linewidth=1.2)
        ax.axvline(shannon_model_db, color="#9467bd", linestyle=":", linewidth=1.3)
    h_extra = [
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.2),
        Line2D([0], [0], color="#9467bd", linestyle=":", linewidth=1.3),
    ]
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles + h_extra,
        labels + [
            f"Theoretical ref. ({SHANNON_REF_DB} dB)",
            f"Shannon ref. at R={RATE:.3f} ({shannon_model_db:.2f} dB)",
        ],
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.1, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bec_ber_fer(res: dict, uncoded_ber: np.ndarray, path: str) -> None:
    cap = capacity_bec(BEC_EPS)
    uncoded_fer = _uncoded_fer_from_ber(uncoded_ber)
    decoders = ["BP", "Min-Sum"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    fig.suptitle(f"({N},{K}) LDPC on BEC: BER/FER vs erasure probability", fontsize=13, fontweight="bold")

    ax = axes[0]
    _plot_decoder_metric(ax, BEC_EPS, res, "ber", decoders)
    ax.semilogy(BEC_EPS, _log_plot_values(uncoded_ber), "k--", linewidth=1.7, label="Uncoded baseline")
    ax.set_xlabel("BEC erasure probability")
    ax.set_ylabel("BER (log scale)")
    ax.set_title("Bit error rate")
    _style_axis(ax)

    ax = axes[1]
    _plot_decoder_metric(ax, BEC_EPS, res, "fer", decoders)
    ax.semilogy(BEC_EPS, _log_plot_values(uncoded_fer), "k--", linewidth=1.7, label="Uncoded baseline")
    ax.set_xlabel("BEC erasure probability")
    ax.set_ylabel("FER (log scale)")
    ax.set_title("Frame error rate")
    _style_axis(ax)

    ax_ref = axes[1].twinx()
    ax_ref.plot(BEC_EPS, cap, color="#555555", linestyle=":", linewidth=1.4, label="Theoretical capacity")
    ax_ref.set_ylabel("Capacity reference (bits/use)", color="#555555")
    ax_ref.tick_params(axis="y", labelcolor="#555555", labelsize=9)
    ax_ref.set_ylim(0.0, 1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = ax_ref.get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_convergence(res_bsc: dict, res_awgn: dict, res_bec: dict, path: str) -> None:
    configs = [
        ("BSC", BSC_P, "p", res_bsc, ["Bit-Flip", "BP", "Min-Sum"]),
        ("AWGN", AWGN_EBNO_DB, "Eb/N0 (dB)", res_awgn, ["BP", "Min-Sum"]),
        ("BEC", BEC_EPS, "erasure probability", res_bec, ["BP", "Min-Sum"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0))
    fig.suptitle(f"({N},{K}) LDPC decoder convergence and iteration cost", fontsize=13, fontweight="bold")

    for col, (name, x, xlabel, res, decoders) in enumerate(configs):
        ax = axes[0, col]
        for decoder in decoders:
            style = DECODER_STYLE[decoder]
            ax.plot(x, res[decoder]["convergence_rate"], label=decoder, color=style["color"], marker=style["marker"], linewidth=1.8, markersize=4.5)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Convergence rate")
        ax.set_ylim(-0.04, 1.04)
        _style_axis(ax)

        ax = axes[1, col]
        for decoder in decoders:
            style = DECODER_STYLE[decoder]
            ax.plot(x, res[decoder]["avg_iterations"], label=decoder, color=style["color"], marker=style["marker"], linewidth=1.8, markersize=4.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average iterations")
        _style_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _coding_gain_series(uncoded_ber: np.ndarray, coded_ber: np.ndarray) -> np.ndarray:
    return np.array([_coding_gain_db(float(u), float(c)) for u, c in zip(uncoded_ber, coded_ber)], dtype=float)


def plot_coding_gain_summary(res_bsc: dict, res_awgn: dict, res_bec: dict, unc_awgn_emp: np.ndarray, path: str) -> None:
    configs = [
        ("BSC", BSC_P, "p", BSC_P.astype(float), res_bsc, ["Bit-Flip", "BP", "Min-Sum"]),
        ("AWGN", AWGN_EBNO_DB, "Eb/N0 (dB)", unc_awgn_emp, res_awgn, ["BP", "Min-Sum"]),
        ("BEC", BEC_EPS, "erasure probability", BEC_EPS.astype(float), res_bec, ["BP", "Min-Sum"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    fig.suptitle("Coding gain summary: 10 log10(uncoded BER / decoder BER)", fontsize=13, fontweight="bold")

    for ax, (name, x, xlabel, uncoded, res, decoders) in zip(axes, configs):
        for decoder in decoders:
            style = DECODER_STYLE[decoder]
            gain = _coding_gain_series(uncoded, res[decoder]["ber"])
            ax.plot(x, gain, label=decoder, color=style["color"], marker=style["marker"], linewidth=1.8, markersize=4.5)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="No gain")
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Coding gain (dB)")
        _style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 0.9])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate(res_bsc: dict, res_awgn: dict, res_bec: dict, unc_awgn_emp: np.ndarray) -> list[str]:
    issues: list[str] = []
    # Compare to simulated uncoded (same Monte Carlo budget, same Eb/N0 definition).
    if not np.any(res_awgn["BP"]["ber"] < unc_awgn_emp):
        bad = np.where(res_awgn["BP"]["ber"] >= unc_awgn_emp)[0]
        issues.append(f"AWGN: BP BER not strictly below uncoded (sim.) at indices {bad.tolist()} (Eb/N0={AWGN_EBNO_DB[bad]})")
    if not np.any(res_awgn["Min-Sum"]["ber"] < unc_awgn_emp):
        bad = np.where(res_awgn["Min-Sum"]["ber"] >= unc_awgn_emp)[0]
        issues.append(f"AWGN: Min-Sum BER not strictly below uncoded (sim.) at indices {bad.tolist()}")

    if not np.all(res_bsc["BP"]["ber"] <= res_bsc["Min-Sum"]["ber"] * 1.01 + 1e-15):
        issues.append("BSC: BP not <= Min-Sum (within 1% tol) at some points")

    if np.allclose(res_bec["BP"]["ber"], res_bec["Min-Sum"]["ber"], rtol=0, atol=1e-18):
        return issues

    return issues


def _result_rows(channel: str, parameter_name: str, parameter_values: np.ndarray, uncoded_ber: np.ndarray, res: dict) -> list[dict]:
    rows = []
    for i, value in enumerate(parameter_values):
        for decoder, metrics in res.items():
            coded_ber = float(metrics["ber"][i])
            rows.append(
                {
                    "channel": channel,
                    "parameter_name": parameter_name,
                    "parameter_value": float(value),
                    "decoder": decoder,
                    "uncoded_BER": float(uncoded_ber[i]),
                    "decoder_BER": coded_ber,
                    "decoder_FER": float(metrics["fer"][i]),
                    "convergence_rate": float(metrics["convergence_rate"][i]),
                    "average_iterations": float(metrics["avg_iterations"][i]),
                    "coding_gain_dB": _coding_gain_db(float(uncoded_ber[i]), coded_ber),
                    "frames_per_point": int(FRAMES_PER_POINT),
                    "N": int(N),
                    "K": int(K),
                    "M": int(M),
                    "rate": float(RATE),
                }
            )
    return rows


def _write_csv(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "channel",
        "parameter_name",
        "parameter_value",
        "decoder",
        "uncoded_BER",
        "decoder_BER",
        "decoder_FER",
        "convergence_rate",
        "average_iterations",
        "coding_gain_dB",
        "frames_per_point",
        "N",
        "K",
        "M",
        "rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compact_metrics(res: dict) -> dict:
    return {
        decoder: {metric: values.tolist() for metric, values in metrics.items()}
        for decoder, metrics in res.items()
    }


def main() -> int:
    print(f"Block length N={N} (target {CODE_LENGTH_TARGET}), K={K}, M={M}, R={RATE:.6f}")
    print(
        f"Frames per point: {FRAMES_PER_POINT}, max_iter BP/MS={MAX_ITER_BP_MS}, "
        f"workers={_pool_workers()} (set EXPERIMENT_WORKERS=1 to disable parallelism)"
    )

    t_run0 = time.perf_counter()
    heavy_phases = 4  # BSC, AWGN, uncoded AWGN sim, BEC (plot/summary is usually small)

    def _phase_done(name: str, t_phase0: float, n_done: int) -> None:
        dt = time.perf_counter() - t_phase0
        total = time.perf_counter() - t_run0
        # Rough ETA: assume remaining heavy phases cost like the average so far.
        if n_done > 0:
            avg = total / n_done
            left = heavy_phases - n_done
            eta = avg * left if left > 0 else 0.0
            print(
                f"  {name} finished in {dt:.1f}s  (elapsed {total:.1f}s, "
                f"rough ETA ~{eta:.0f}s left if phases stay similar)",
                flush=True,
            )
        else:
            print(f"  {name} finished in {dt:.1f}s  (elapsed {total:.1f}s)", flush=True)

    print("BSC sweep...", flush=True)
    t0 = time.perf_counter()
    res_bsc = run_bsc(BSC_P)
    _phase_done("BSC", t0, 1)

    print("AWGN sweep...", flush=True)
    t0 = time.perf_counter()
    res_awgn = run_awgn(AWGN_EBNO_DB)
    _phase_done("AWGN", t0, 2)

    print("AWGN uncoded (sim.)...", flush=True)
    t0 = time.perf_counter()
    unc_awgn_emp = run_awgn_uncoded_empirical(AWGN_EBNO_DB)
    _phase_done("AWGN uncoded", t0, 3)

    print("BEC sweep...", flush=True)
    t0 = time.perf_counter()
    res_bec = run_bec(BEC_EPS)
    _phase_done("BEC", t0, 4)

    unc_bsc = BSC_P.astype(float)
    unc_bec = BEC_EPS.astype(float)
    figure_files = {
        "bsc_ber_fer": os.path.join(FIGURES_DIR, "bsc_ber_fer.png"),
        "awgn_ber_fer": os.path.join(FIGURES_DIR, "awgn_ber_fer.png"),
        "bec_ber_fer": os.path.join(FIGURES_DIR, "bec_ber_fer.png"),
        "decoder_convergence": os.path.join(FIGURES_DIR, "decoder_convergence.png"),
        "coding_gain_summary": os.path.join(FIGURES_DIR, "coding_gain_summary.png"),
    }
    t0 = time.perf_counter()
    plot_bsc_ber_fer(res_bsc, unc_bsc, figure_files["bsc_ber_fer"])
    plot_awgn_ber_fer(res_awgn, unc_awgn_emp, figure_files["awgn_ber_fer"])
    plot_bec_ber_fer(res_bec, unc_bec, figure_files["bec_ber_fer"])
    plot_decoder_convergence(res_bsc, res_awgn, res_bec, figure_files["decoder_convergence"])
    plot_coding_gain_summary(res_bsc, res_awgn, res_bec, unc_awgn_emp, figure_files["coding_gain_summary"])
    print(
        f"  Plots saved in {time.perf_counter() - t0:.1f}s  (total run {time.perf_counter() - t_run0:.1f}s)",
        flush=True,
    )
    print("Saved:", *figure_files.values(), sep="\n  ")

    issues = validate(res_bsc, res_awgn, res_bec, unc_awgn_emp)
    unc_awgn = _ber_uncoded_awgn_bpsk_db(AWGN_EBNO_DB)

    bsc_csv = os.path.join(TABLES_DIR, "bsc_results.csv")
    awgn_csv = os.path.join(TABLES_DIR, "awgn_results.csv")
    bec_csv = os.path.join(TABLES_DIR, "bec_results.csv")
    _write_csv(bsc_csv, _result_rows("BSC", "p", BSC_P, unc_bsc, res_bsc))
    _write_csv(awgn_csv, _result_rows("AWGN", "EbN0_dB", AWGN_EBNO_DB, unc_awgn_emp, res_awgn))
    _write_csv(bec_csv, _result_rows("BEC", "epsilon", BEC_EPS, unc_bec, res_bec))
    print("Wrote:", bsc_csv, awgn_csv, bec_csv, sep="\n  ")

    summary = {
        "N": int(N),
        "K": int(K),
        "M": int(M),
        "RATE": float(RATE),
        "CODE_LENGTH_TARGET": int(CODE_LENGTH_TARGET),
        "frames_per_point": int(FRAMES_PER_POINT),
        "max_iter_bp_minsum": int(MAX_ITER_BP_MS),
        "max_iter_bit_flip": int(MAX_ITER_BF),
        "table_files": {
            "bsc": bsc_csv,
            "awgn": awgn_csv,
            "bec": bec_csv,
        },
        "figure_files": figure_files,
        "bsc_p": BSC_P.tolist(),
        "bsc_uncoded_ber": unc_bsc.tolist(),
        "bsc_metrics": _compact_metrics(res_bsc),
        "awgn_ebno_db": AWGN_EBNO_DB.tolist(),
        "ber_uncoded_awgn_analytic": unc_awgn.tolist(),
        "ber_uncoded_awgn_simulated": unc_awgn_emp.tolist(),
        "awgn_metrics": _compact_metrics(res_awgn),
        "bec_eps": BEC_EPS.tolist(),
        "bec_uncoded_ber": unc_bec.tolist(),
        "bec_metrics": _compact_metrics(res_bec),
        "coding_gain_formula": "10 * log10(uncoded_BER / decoder_BER)",
        "bec_note": "BP and Min-Sum are both evaluated with the supported LLR decoders; matching curves are reported as matching values.",
        "validation_issues": issues,
    }
    jp = os.path.join(SUMMARIES_DIR, "experiment_summary.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", jp)

    if issues:
        print("\nValidation notes (review if unexpected):")
        for s in issues:
            print("  -", s)
    else:
        print("\nValidation: all checks passed.")

    return 0 if not issues else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n!!! experiment.py FAILED !!!", file=sys.stderr)
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from exc
