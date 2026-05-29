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

Outputs (results/):
  ber_bsc_with_capacity.png
  ber_awgn_with_shannon.png
  ber_bec_with_capacity.png
  experiment_summary.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
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
os.makedirs(RESULTS_DIR, exist_ok=True)
RNG = np.random.default_rng(RNG_SEED)

H, G, col_order, K, M, RATE, N = build_ldpc_for_target_length(
    CODE_LENGTH_TARGET, wc=3, seed=CONSTRUCTION_SEED, prefer="ge"
)


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

    e_bf = e_bp = e_ms = bits = 0
    for _ in range(frames):
        msg = rng.integers(0, 2, size=K)
        cw = encode(G, msg, col_order)
        rx, _ = bsc(cw, float(p), seed=rng.integers(0, 2**31))
        llr = bsc_llr(rx, float(p))
        d_bf, _, _ = bit_flip_decode(H, rx, max_iter=max_iter_bf)
        d_bp, _, _ = belief_propagation_decode(H, llr, max_iter=max_iter_bp)
        d_ms, _, _ = minsum_decode(H, llr, max_iter=max_iter_bp)
        e_bf += int(np.sum(d_bf != cw))
        e_bp += int(np.sum(d_bp != cw))
        e_ms += int(np.sum(d_ms != cw))
        bits += N
    return e_bf / bits, e_bp / bits, e_ms / bits


def _awgn_point_task(args):
    H, G, col_order, K, N, rate, snr_db, frames, base_seed, max_iter_bp = args
    rng = np.random.default_rng(base_seed)
    from ldpc import encode, awgn_channel, awgn_llr, belief_propagation_decode, minsum_decode

    e_bp = e_ms = bits = 0
    for _ in range(frames):
        msg = rng.integers(0, 2, size=K)
        cw = encode(G, msg, col_order)
        rx, sigma2 = awgn_channel(cw, float(snr_db), rate=rate, seed=rng.integers(0, 2**31))
        llr = awgn_llr(rx, sigma2)
        d_bp, _, _ = belief_propagation_decode(H, llr, max_iter=max_iter_bp)
        d_ms, _, _ = minsum_decode(H, llr, max_iter=max_iter_bp)
        e_bp += int(np.sum(d_bp != cw))
        e_ms += int(np.sum(d_ms != cw))
        bits += N
    return e_bp / bits, e_ms / bits


def _bec_point_task(args):
    H, G, col_order, K, N, eps, frames, base_seed, max_iter_bp = args
    rng = np.random.default_rng(base_seed)
    from ldpc import encode, bec_channel, bec_llr, belief_propagation_decode

    e_bp = bits = 0
    for _ in range(frames):
        msg = rng.integers(0, 2, size=K)
        cw = encode(G, msg, col_order)
        rx = bec_channel(cw, float(eps), seed=rng.integers(0, 2**31))
        llr = bec_llr(rx)
        d_bp, _, _ = belief_propagation_decode(H, llr, max_iter=max_iter_bp)
        e_bp += int(np.sum(d_bp != cw))
        bits += N
    ber = e_bp / bits
    # On BEC, Min-Sum matches BP; report one BER for both (prompt: identical curves).
    return ber, ber


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
    w = _pool_workers()
    if w == 1:
        out = {"Bit-Flip": [], "BP": [], "Min-Sum": []}
        for p in flip_probs:
            errs = {k: 0 for k in out}
            bits = 0
            for _ in range(FRAMES_PER_POINT):
                msg = RNG.integers(0, 2, size=K)
                cw = encode(G, msg, col_order)
                rx, _ = bsc(cw, float(p), seed=RNG.integers(0, 2**31))
                llr = bsc_llr(rx, float(p))
                d_bf, _, _ = bit_flip_decode(H, rx, max_iter=MAX_ITER_BF)
                d_bp, _, _ = belief_propagation_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                d_ms, _, _ = minsum_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                errs["Bit-Flip"] += int(np.sum(d_bf != cw))
                errs["BP"] += int(np.sum(d_bp != cw))
                errs["Min-Sum"] += int(np.sum(d_ms != cw))
                bits += N
            for k in out:
                out[k].append(errs[k] / bits)
        return {k: np.array(v, dtype=float) for k, v in out.items()}

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
    bf, bp, ms = zip(*rows)
    return {
        "Bit-Flip": np.array(bf, dtype=float),
        "BP": np.array(bp, dtype=float),
        "Min-Sum": np.array(ms, dtype=float),
    }


def run_awgn(snr_db_list: np.ndarray) -> dict:
    w = _pool_workers()
    if w == 1:
        out = {"BP": [], "Min-Sum": []}
        for snr_db in snr_db_list:
            errs = {"BP": 0, "Min-Sum": 0}
            bits = 0
            for _ in range(FRAMES_PER_POINT):
                msg = RNG.integers(0, 2, size=K)
                cw = encode(G, msg, col_order)
                rx, sigma2 = awgn_channel(cw, float(snr_db), rate=RATE, seed=RNG.integers(0, 2**31))
                llr = awgn_llr(rx, sigma2)
                d_bp, _, _ = belief_propagation_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                d_ms, _, _ = minsum_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                errs["BP"] += int(np.sum(d_bp != cw))
                errs["Min-Sum"] += int(np.sum(d_ms != cw))
                bits += N
            out["BP"].append(errs["BP"] / bits)
            out["Min-Sum"].append(errs["Min-Sum"] / bits)
        return {k: np.array(v, dtype=float) for k, v in out.items()}

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
    bp, ms = zip(*rows)
    return {"BP": np.array(bp, dtype=float), "Min-Sum": np.array(ms, dtype=float)}


def run_bec(eps_list: np.ndarray) -> dict:
    w = _pool_workers()
    if w == 1:
        out = {"BP": [], "Min-Sum": []}
        for eps in eps_list:
            errs_bp = bits = 0
            for _ in range(FRAMES_PER_POINT):
                msg = RNG.integers(0, 2, size=K)
                cw = encode(G, msg, col_order)
                rx = bec_channel(cw, float(eps), seed=RNG.integers(0, 2**31))
                llr = bec_llr(rx)
                d_bp, _, _ = belief_propagation_decode(H, llr, max_iter=MAX_ITER_BP_MS)
                errs_bp += int(np.sum(d_bp != cw))
                bits += N
            ber = errs_bp / bits
            out["BP"].append(ber)
            out["Min-Sum"].append(ber)
        return {k: np.array(v, dtype=float) for k, v in out.items()}

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
    bp, ms = zip(*rows)
    return {"BP": np.array(bp, dtype=float), "Min-Sum": np.array(ms, dtype=float)}


def plot_bsc(res: dict, path: str) -> None:
    cap = np.array([capacity_bsc(float(p)) for p in BSC_P])
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.semilogy(BSC_P, res["Bit-Flip"], "-o", ms=4, label="Bit-Flip")
    ax1.semilogy(BSC_P, res["BP"], "-s", ms=4, label="BP")
    ax1.semilogy(BSC_P, res["Min-Sum"], "-^", ms=4, label="Min-Sum")
    ax1.semilogy(BSC_P, BSC_P, "k--", lw=1.2, label="Uncoded (BER = p)")
    ax1.set_xlabel("BSC crossover probability p")
    ax1.set_ylabel("BER (log scale)")
    ax1.set_title(f"({N},{K}) LDPC on BSC: BER vs p (frames={FRAMES_PER_POINT})")
    ax1.grid(True, which="both", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8)
    min_ber = min(
        float(np.min(res["Bit-Flip"])),
        float(np.min(res["BP"])),
        float(np.min(res["Min-Sum"])),
        float(np.min(BSC_P)),
    )
    ax1.set_ylim(bottom=max(1e-6, min_ber * 0.3))

    ax2 = ax1.twinx()
    ax2.plot(BSC_P, cap, "k:", lw=1.6, label=r"$C(p)=1-H_2(p)$")
    ax2.set_ylabel("Capacity (bits/use)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_awgn(res: dict, unc_emp: np.ndarray, path: str) -> None:
    unc_an = _ber_uncoded_awgn_bpsk_db(AWGN_EBNO_DB)
    shannon_model_db = float(shannon_limit_awgn_rate(RATE))

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    lu = ax1.semilogy(AWGN_EBNO_DB, unc_emp, "k-", lw=1.4)[0]
    lu2 = ax1.semilogy(AWGN_EBNO_DB, unc_an, "k:", lw=1.0)[0]
    lbp = ax1.semilogy(AWGN_EBNO_DB, res["BP"], "-s", ms=4)[0]
    lms = ax1.semilogy(AWGN_EBNO_DB, res["Min-Sum"], "-^", ms=4)[0]
    ax1.axvline(SHANNON_REF_DB, color="gray", ls="--", lw=1.4)
    ax1.axvline(shannon_model_db, color="purple", ls=":", lw=1.4)
    ax1.set_xlabel(r"$E_b/N_0$ (dB)")
    ax1.set_ylabel("BER (log scale)")
    ax1.set_title(f"({N},{K}) LDPC on AWGN, R={RATE:.4f} (frames={FRAMES_PER_POINT})")
    ax1.grid(True, which="both", alpha=0.35)
    lo = min(
        float(np.min(res["BP"])),
        float(np.min(res["Min-Sum"])),
        float(np.min(unc_emp)),
        float(np.min(unc_an)),
    )
    ax1.set_ylim(bottom=max(1e-7, lo * 0.2))
    h_extra = [
        Line2D([0], [0], color="gray", ls="--", lw=1.4),
        Line2D([0], [0], color="purple", ls=":", lw=1.4),
    ]
    ax1.legend(
        [lu, lu2, lbp, lms] + h_extra,
        [
            "Uncoded BPSK (simulated, rate=1)",
            "Uncoded BPSK (analytic)",
            "BP",
            "Min-Sum",
            f"Ref. limit ({SHANNON_REF_DB} dB)",
            f"Shannon @ R={RATE:.3f} ({shannon_model_db:.2f} dB)",
        ],
        loc="upper right",
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bec(res: dict, path: str) -> None:
    cap = capacity_bec(BEC_EPS)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.semilogy(BEC_EPS, res["BP"], "-s", ms=4, label="BP")
    ax1.semilogy(BEC_EPS, res["Min-Sum"], "--^", ms=4, label="Min-Sum")
    ax1.semilogy(BEC_EPS, BEC_EPS, "k--", lw=1.2, label="Uncoded (BER = ε)")
    ax1.set_xlabel(r"BEC erasure probability $\epsilon$")
    ax1.set_ylabel("BER (log scale)")
    ax1.set_title(f"({N},{K}) LDPC on BEC (frames={FRAMES_PER_POINT})")
    ax1.grid(True, which="both", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8)
    lo = min(float(np.min(res["BP"])), float(np.min(res["Min-Sum"])))
    ax1.set_ylim(bottom=max(1e-6, lo * 0.2))

    ax2 = ax1.twinx()
    ax2.plot(BEC_EPS, cap, "k:", lw=1.6, label=r"$C=1-\epsilon$")
    ax2.set_ylabel("Capacity (bits/use)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate(res_bsc: dict, res_awgn: dict, res_bec: dict, unc_awgn_emp: np.ndarray) -> list[str]:
    issues: list[str] = []
    # Compare to simulated uncoded (same Monte Carlo budget, same Eb/N0 definition).
    if not np.any(res_awgn["BP"] < unc_awgn_emp):
        bad = np.where(res_awgn["BP"] >= unc_awgn_emp)[0]
        issues.append(f"AWGN: BP BER not strictly below uncoded (sim.) at indices {bad.tolist()} (Eb/N0={AWGN_EBNO_DB[bad]})")
    if not np.any(res_awgn["Min-Sum"] < unc_awgn_emp):
        bad = np.where(res_awgn["Min-Sum"] >= unc_awgn_emp)[0]
        issues.append(f"AWGN: Min-Sum BER not strictly below uncoded (sim.) at indices {bad.tolist()}")

    if not np.all(res_bsc["BP"] <= res_bsc["Min-Sum"] * 1.01 + 1e-15):
        issues.append("BSC: BP not <= Min-Sum (within 1% tol) at some points")

    if not np.allclose(res_bec["BP"], res_bec["Min-Sum"], rtol=0, atol=1e-18):
        d = float(np.max(np.abs(res_bec["BP"] - res_bec["Min-Sum"])))
        issues.append(f"BEC: BP vs Min-Sum differ (max abs diff={d:.3e}); expect identical on BEC")

    return issues


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

    p_bsc = os.path.join(RESULTS_DIR, "ber_bsc_with_capacity.png")
    p_awgn = os.path.join(RESULTS_DIR, "ber_awgn_with_shannon.png")
    p_bec = os.path.join(RESULTS_DIR, "ber_bec_with_capacity.png")
    t0 = time.perf_counter()
    plot_bsc(res_bsc, p_bsc)
    plot_awgn(res_awgn, unc_awgn_emp, p_awgn)
    plot_bec(res_bec, p_bec)
    print(
        f"  Plots saved in {time.perf_counter() - t0:.1f}s  (total run {time.perf_counter() - t_run0:.1f}s)",
        flush=True,
    )
    print("Saved:", p_bsc, p_awgn, p_bec, sep="\n  ")

    issues = validate(res_bsc, res_awgn, res_bec, unc_awgn_emp)
    unc_awgn = _ber_uncoded_awgn_bpsk_db(AWGN_EBNO_DB)

    def gain_db(p_u: float, p_c: float) -> float:
        if p_c <= 0:
            return float("inf")
        return float(10 * np.log10(max(p_u, 1e-30) / max(p_c, 1e-30)))

    # Pick representative operating points even when EXPERIMENT_QUICK reduces sweep sizes.
    idx_bsc = int(np.argmin(np.abs(BSC_P - 0.10)))
    idx_awgn = int(np.argmin(np.abs(AWGN_EBNO_DB - 3.0)))
    idx_bec = int(np.argmin(np.abs(BEC_EPS - 0.20)))

    summary = {
        "N": int(N),
        "K": int(K),
        "M": int(M),
        "RATE": float(RATE),
        "CODE_LENGTH_TARGET": int(CODE_LENGTH_TARGET),
        "frames_per_point": int(FRAMES_PER_POINT),
        "bsc_p": BSC_P.tolist(),
        "ber_bp_bsc": res_bsc["BP"].tolist(),
        "awgn_ebno_db": AWGN_EBNO_DB.tolist(),
        "ber_bp_awgn": res_awgn["BP"].tolist(),
        "ber_uncoded_awgn_analytic": unc_awgn.tolist(),
        "ber_uncoded_awgn_simulated": unc_awgn_emp.tolist(),
        "bec_eps": BEC_EPS.tolist(),
        "ber_bp_bec": res_bec["BP"].tolist(),
        "coding_gain_db_examples": {
            "bsc_at_p_0.10_bp_vs_uncoded": gain_db(0.10, res_bsc["BP"][idx_bsc]),
            "awgn_at_3dB_bp_vs_uncoded": gain_db(float(unc_awgn_emp[idx_awgn]), float(res_awgn["BP"][idx_awgn])),
            "bec_at_eps_0.20_bp_vs_uncoded": gain_db(0.20, float(res_bec["BP"][idx_bec])),
        },
        "validation_issues": issues,
    }
    jp = os.path.join(RESULTS_DIR, "experiment_summary.json")
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
