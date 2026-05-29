# NOTE:
# This demo uses a small (20,10) LDPC code for visualization.
# Performance is limited and not representative of practical LDPC codes.
# See experiment.py for realistic performance evaluation.

#!/usr/bin/env python3
"""
Simple LDPC Demo - end-to-end in one file.

Default: (20,10) from ldpc.py (matrices printed in full). For large-n construction
parameters used by experiment.py, run:  python simple_demo.py --large-info

Supports bit-flipping (hard), Belief Propagation (soft), and Min-Sum (soft).
Run:  python simple_demo.py
"""

import argparse
import numpy as np
from ldpc import (
    build_H,
    full_rank_H,
    make_generator,
    encode,
    bsc,
    bsc_llr,
    bit_flip_decode,
    belief_propagation_decode,
    belief_propagation_decode_with_history,
    minsum_decode,
)


# =====================================================================
#  Main demo
# =====================================================================

def show_bits(arr, label="", width=None):
    s = "".join(str(b) for b in arr)
    if width and len(s) > width:
        s = s[:width] + "..."
    print(f"  {label:>18s}: {s}")


def main():
    np.set_printoptions(linewidth=120)

    n = 20        # block length
    wc, wr = 3, 5
    flip_prob = 0.15
    seed = 7

    H_full = build_H(n, wc, wr, seed=seed)
    H = full_rank_H(H_full)  # (20,10) code: 10 x 20, syndrome length m = 10
    m, n = H.shape[0], H.shape[1]

    print("=" * 60)
    print("  LDPC Codes - Simple End-to-End Demo")
    print("=" * 60)

    # --- 1. Build H ---
    print(f"\n1) Parity-check matrix H  (m x n = {m} x {n}), syndrome length = {m}")
    print(f"   Column weight wc={wc}, row weight wr={wr}")
    print(f"\n   H =")
    for row in H:
        print("   " + " ".join(str(x) for x in row))

    # --- 2. Generator matrix ---
    G, col_order, k = make_generator(H)
    print(f"\n2) Generator matrix G  ({k} x {n})")
    print(f"   Code rate R = k/n = {k}/{n} = {k/n:.2f}")
    print(f"\n   G =")
    for row in G:
        print("   " + " ".join(str(x) for x in row))

    # --- 3. Encode ---
    rng = np.random.default_rng(seed)
    message = rng.integers(0, 2, size=k)
    codeword = encode(G, message, col_order)

    print(f"\n3) Encode a random message of {k} bits")
    show_bits(message, "message")
    show_bits(codeword, "codeword")

    syndrome = (H @ codeword) % 2
    print(f"   H @ codeword mod 2 = {'0' * m}  (valid codeword: {not syndrome.any()})")

    # --- 4. Transmit over BSC ---
    received, error_vec = bsc(codeword, flip_prob, seed=seed + 1)
    num_flipped = error_vec.sum()

    print(f"\n4) Transmit over BSC  (flip probability = {flip_prob})")
    show_bits(codeword, "sent")
    show_bits(error_vec, "error pattern")
    show_bits(received, "received")
    print(f"   Bits flipped by channel: {num_flipped}/{n}")

    syndrome = (H @ received) % 2
    print(f"   Syndrome (H @ received mod 2): {''.join(str(s) for s in syndrome)}"
          f"  (all zero = {not syndrome.any()})")

    # --- 5. Decode ---
    print(f"\n5) Decode (bit-flipping, max 20 iterations)")
    decoded, converged, iters = bit_flip_decode(H, received, max_iter=20)

    show_bits(received, "received")
    show_bits(decoded, "decoded")
    show_bits(codeword, "original")

    bit_errors = np.sum(decoded != codeword)
    print(f"\n   Converged: {converged}")
    print(f"   Iterations used: {iters}")
    print(f"   Bit errors after decoding: {bit_errors}/{n}")

    msg_errors = np.sum(
        decoded[col_order[:k]] != message
    )
    print(f"   Message errors: {msg_errors}/{k}")

    # --- 6. Message-passing steps (BP) ---
    print(f"\n6) Message-passing iterations (Belief Propagation, same received vector)")
    print(f"   Each row: iteration | syndrome (m={m} bits) | tentative decoded (n={n} bits) | LLR sign (+ = likely 0) | errors vs codeword")
    llr_demo = bsc_llr(received, flip_prob)
    _, _, iters_bp, hist = belief_propagation_decode_with_history(H, llr_demo, max_iter=25)
    print(f"   {'iter':>4}  {'syndrome':>12}  {'decoded':>24}  {'LLR sign':>24}  err")
    print(f"   {'-'*4}  {'-'*12}  {'-'*24}  {'-'*24}  ---")
    for h in hist:
        syn_str = "".join(str(s) for s in h["syndrome"])
        dec_str = "".join(str(b) for b in h["decoded"])
        llr_sign = "".join("+" if h["total_llr"][i] >= 0 else "-" for i in range(n))
        err = np.sum(h["decoded"] != codeword)
        print(f"   {h['iter']:>4}  {syn_str:>12}  {dec_str:>24}  {llr_sign:>24}  {err:>3}")
    print(f"   -> Converged: {np.all(hist[-1]['syndrome'] == 0)}, total iterations: {len(hist)}")
    print(f"   BP behavior depends on channel conditions and LLR quality. It may converge faster or slower depending on noise realization.")

    # --- 7. Summary ---
    print(f"\n{'=' * 60}")
    if bit_errors == 0:
        print("   SUCCESS — all channel errors corrected by the decoder!")
    else:
        print(f"   FAILURE — {bit_errors} bit(s) still wrong")
    print("=" * 60)

    # --- 8. Quick Monte Carlo ---
    print(f"\n\n--- Bonus: Quick BER test (500 frames) ---\n")
# NOTE: Uses bit-flipping only; performance is limited compared to BP.\n")
    print(f"  {'flip_prob':>10s}  {'BER':>10s}  {'frame_err_rate':>14s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*14}")

    for fp in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        frames = 500
        total_errs = 0
        frame_errs = 0
        total_bits = 0
        for f in range(frames):
            msg = rng.integers(0, 2, size=k)
            cw = encode(G, msg, col_order)
            rx, _ = bsc(cw, fp, seed=f)
            dec, conv, _ = bit_flip_decode(H, rx)
            errs = np.sum(dec != cw)
            total_errs += errs
            total_bits += n
            if errs > 0:
                frame_errs += 1
        ber = total_errs / total_bits
        fer = frame_errs / frames
        print(f"  {fp:>10.2f}  {ber:>10.4e}  {fer:>14.3f}")


def interactive_mode():
    """Let the user type a message, choose which bits to flip, and watch decoding."""

    n = 20
    wc, wr = 3, 5
    seed = 7
    H_full = build_H(n, wc, wr, seed=seed)
    H = full_rank_H(H_full)
    G, col_order, k = make_generator(H)
    m = H.shape[0]

    print("\n" + "=" * 60)
    print("  LDPC Interactive Mode")
    print("=" * 60)
    print(f"\n  Code parameters: n={n}, k={k}, rate={k/n:.2f}")
    print(f"  You provide a {k}-bit message and choose which")
    print(f"  codeword bits to corrupt. The decoder tries to fix them.\n")

    while True:
        # --- get message from user ---
        print("-" * 60)
        raw = input(f"  Enter {k} message bits (0s and 1s), or 'r' for random, 'q' to quit:\n  > ").strip()

        if raw.lower() == "q":
            print("\n  Bye!")
            break

        if raw.lower() == "r":
            message = np.random.randint(0, 2, size=k)
            print(f"  Random message generated.")
        else:
            bits = [int(c) for c in raw if c in "01"]
            if len(bits) != k:
                print(f"  Need exactly {k} bits, got {len(bits)}. Try again.")
                continue
            message = np.array(bits, dtype=int)

        # --- encode ---
        codeword = encode(G, message, col_order)
        syndrome = (H @ codeword) % 2

        print()
        show_bits(message, "your message")
        show_bits(codeword, "codeword")
        print(f"  {'syndrome':>18s}: {''.join(str(s) for s in syndrome)}  (valid: {not syndrome.any()})")

        # --- get error positions from user ---
        print(f"\n  Codeword has {n} bits (positions 0-{n-1}).")
        print(f"  Bit positions:      {''.join(str(i % 10) for i in range(n))}")
        show_bits(codeword, "codeword")
        pos_raw = input("  Which bit positions to flip? (e.g. '2 7 15', or 'none'):\n  > ").strip()

        if pos_raw.lower() in ("none", ""):
            flip_positions = []
        else:
            try:
                flip_positions = [int(x) for x in pos_raw.split()]
                bad = [p for p in flip_positions if p < 0 or p >= n]
                if bad:
                    print(f"  Positions {bad} are out of range (0-{n-1}). Try again.")
                    continue
            except ValueError:
                print("  Could not parse positions. Try again.")
                continue

        # --- apply errors ---
        received = codeword.copy()
        error_vec = np.zeros(n, dtype=int)
        for p in flip_positions:
            received[p] ^= 1
            error_vec[p] = 1

        syndrome = (H @ received) % 2
        has_errors = syndrome.any()

        print()
        show_bits(codeword, "original")
        show_bits(error_vec, "error pattern")
        show_bits(received, "corrupted")
        print(f"  {'bits flipped':>18s}: {len(flip_positions)}")
        print(f"  {'syndrome':>18s}: {''.join(str(s) for s in syndrome)}")

        if not has_errors:
            print("\n  Syndrome is all zeros — no errors detected!")
            if len(flip_positions) == 0:
                print("  (You didn't flip any bits, so that's expected.)")
            else:
                print("  WARNING: errors went undetected (decoded to wrong codeword).")
            continue

        print(f"\n  Syndrome is non-zero — errors detected! Running decoder...\n")

        # --- decode with verbose output ---
        decoded, converged, iters = bit_flip_decode_verbose(H, received)

        show_bits(received, "corrupted")
        show_bits(decoded, "decoded")
        show_bits(codeword, "original")

        bit_errors = np.sum(decoded != codeword)
        msg_decoded = decoded[col_order[:k]]
        msg_errors = np.sum(msg_decoded != message)

        print()
        if bit_errors == 0:
            print(f"  SUCCESS — all {len(flip_positions)} error(s) corrected "
                  f"in {iters} iteration(s)!")
        else:
            print(f"  FAILURE — {bit_errors} bit(s) still wrong after {iters} iterations.")
            print(f"  Message errors: {msg_errors}/{k}")
        print()


def bit_flip_decode_verbose(H, received, max_iter=20):
    """Bit-flipping decoder with per-iteration output."""
    c = received.copy()
    m, n = H.shape

    for it in range(1, max_iter + 1):
        syndrome = (H @ c) % 2
        num_unsat = int(syndrome.sum())

        if not syndrome.any():
            print(f"    Iter {it:>2d}: syndrome = all zeros  ->  DONE")
            return c, True, it

        unsat_per_bit = H.T @ syndrome
        worst = int(np.argmax(unsat_per_bit))
        worst_count = int(unsat_per_bit[worst])

        print(f"    Iter {it:>2d}: {num_unsat} unsatisfied checks  "
              f"->  flip bit {worst} (in {worst_count} failing checks)")
        c[worst] ^= 1

    final_ok = np.all((H @ c) % 2 == 0)
    if final_ok:
        print(f"    Iter {max_iter:>2d}: syndrome = all zeros  ->  DONE")
    else:
        print(f"    Reached max iterations ({max_iter}) without converging.")
    return c, final_ok, max_iter


def print_large_code_info():
    from ldpc_construction import build_ldpc_for_target_length

    for target, label in [(200, "n ~ 200 (target)"), (500, "n ~ 500 (target)")]:
        H, G, col_order, k, m, rate, n = build_ldpc_for_target_length(
            target, wc=3, seed=7, prefer="ge"
        )
        print(f"\n{label}: built (n,k)=({n},{k}), m={m}, R={rate:.6f}")
        print(f"  H shape {H.shape}, density {H.mean():.4f}")
        print(f"  (Matrices omitted; use experiment.py for BER plots.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LDPC simple demo (default n=20).")
    ap.add_argument(
        "--large-info",
        action="store_true",
        help="Print (n,k) for target lengths 200 and 500 from ldpc_construction, then exit.",
    )
    ap.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run the scripted demo only; skip interactive session.",
    )
    args = ap.parse_args()
    if args.large_info:
        print_large_code_info()
    else:
        main()
        if not args.no_interactive:
            interactive_mode()
