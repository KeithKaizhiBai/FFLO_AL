from __future__ import annotations

from ml_phase.exact_oracle import (
    cluster_branch_candidates,
    mark_energy_window_pruning,
    select_local_refine_targets,
)


def _candidate(
    rank: int,
    q: float,
    delta: float,
    energy_above: float,
    *,
    edge_risk: bool = False,
    low_energy: bool = False,
) -> dict[str, object]:
    return {
        "minimum_rank": rank,
        "grid_index": rank,
        "q_local_min": q,
        "Delta_local_min": delta,
        "DeltaF_local_min": -1.0e-3 + energy_above,
        "energy_above_global": energy_above,
        "distance_to_q_edge": 0.01 if edge_risk else 0.2,
        "within_low_energy_window": low_energy,
        "edge_risk": edge_risk,
    }


def test_mandatory_branch_keep_survives_clustering_pruning_and_selection():
    rows = [
        _candidate(1, 0.00, 0.250, 0.0),
        _candidate(2, 0.40, 0.250, 5.0e-4, edge_risk=True),
        _candidate(3, 0.80, 0.002, 6.0e-4),
        _candidate(4, 1.20, 0.350, 7.0e-4, low_energy=True),
        _candidate(5, 1.60, 0.350, 8.0e-4),
    ]

    clustered = cluster_branch_candidates(
        rows,
        coarse_dq=0.004,
        coarse_dDelta=0.002,
        numerical_energy_scale=1.0e-5,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
    )
    marked = mark_energy_window_pruning(
        clustered,
        local_refine_energy_window=1.0e-4,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        enabled=True,
    )
    selected = select_local_refine_targets(
        marked,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        max_total_refined_basins=2,
        enable_selective_refinement=True,
        max_optional_refined_basins=0,
        mandatory_basins_can_exceed_cap=True,
    )

    selected_ids = {int(row["minimum_rank"]) for row in selected}
    assert selected_ids == {1, 2, 3, 4}

    pruned_ids = {int(row["minimum_rank"]) for row in marked if bool(row["pruned_by_energy_window"])}
    assert pruned_ids == {5}

    reasons_by_id = {
        int(row["minimum_rank"]): set(str(row["mandatory_basin_reasons"]).split(";"))
        for row in marked
        if bool(row["mandatory_basin"])
    }
    assert "global_best" in reasons_by_id[1]
    assert "edge_risk" in reasons_by_id[2]
    assert "Delta_near_epsilon" in reasons_by_id[3]
    assert "near_degenerate" in reasons_by_id[4]


def test_mandatory_branch_keep_strict_cap_is_explicitly_configured():
    rows = [
        _candidate(1, 0.00, 0.250, 0.0),
        _candidate(2, 0.40, 0.250, 5.0e-4, edge_risk=True),
        _candidate(3, 0.80, 0.002, 6.0e-4),
    ]

    selected = select_local_refine_targets(
        rows,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        max_total_refined_basins=2,
        enable_selective_refinement=True,
        max_optional_refined_basins=0,
        mandatory_basins_can_exceed_cap=False,
    )

    assert [int(row["minimum_rank"]) for row in selected] == [1, 2]
