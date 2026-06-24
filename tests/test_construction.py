from ldpc_construction import analyze_H, build_ldpc_for_target_length


def test_target_200_resolves_to_204():
    H, G, col_order, K, M, rate, N = build_ldpc_for_target_length(
        200, wc=3, seed=42, prefer="ge"
    )

    assert N == 204
    assert K == 104
    assert M == 100
    assert H.shape == (M, N)
    assert G.shape == (K, N)
    assert len(col_order) == N
    assert abs(rate - 0.5) < 0.02


def test_analyze_h_returns_required_fields(ldpc_code):
    diagnostics = analyze_H(ldpc_code["H"])
    required = {
        "M",
        "N",
        "rank_gf2",
        "K",
        "rate",
        "density",
        "row_weight_min",
        "row_weight_max",
        "row_weight_mean",
        "column_weight_min",
        "column_weight_max",
        "column_weight_mean",
        "estimated_4_cycles",
    }

    assert required.issubset(diagnostics)
    assert diagnostics["N"] == 204
    assert diagnostics["K"] == 104
    assert diagnostics["rank_gf2"] == 100
    assert 0 < diagnostics["density"] < 1
