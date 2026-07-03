from __future__ import annotations

from ml_phase.exact_oracle import (
    _local_refine_selection_reason,
    cluster_branch_candidates,
    mark_energy_window_pruning,
    select_local_refine_targets,
)


def _candidate(
    rank: int,
    q: float,
    delta: float,
    deltaf: float,
    *,
    edge_risk: bool = False,
    low_energy: bool = False,
) -> dict[str, object]:
    return {
        "minimum_rank": rank,
        "grid_index": rank * 10,
        "q_local_min": q,
        "Delta_local_min": delta,
        "DeltaF_local_min": deltaf,
        "energy_above_global": deltaf + 1.0e-3,
        "distance_to_q_edge": 0.01 if edge_risk else 0.2,
        "within_low_energy_window": low_energy,
        "edge_risk": edge_risk,
    }


def test_basin_clustering_merges_duplicate_coarse_minima():
    rows = [
        _candidate(1, 0.100, 0.200, -1.0000e-3, low_energy=True),
        _candidate(2, 0.104, 0.201, -0.9995e-3, low_energy=True),
        _candidate(3, 0.450, 0.300, -0.7000e-3),
    ]

    clustered = cluster_branch_candidates(
        rows,
        coarse_dq=0.004,
        coarse_dDelta=0.002,
        numerical_energy_scale=1.0e-5,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.03,
    )

    assert len(clustered) == 2
    assert clustered[0]["cluster_size"] == 2
    assert clustered[0]["merged_branch_ids"] == "1;2"
    assert clustered[0]["cluster_reason"] == "q_delta_energy_duplicate"
    assert clustered[1]["cluster_size"] == 1


def test_basin_clustering_preserves_mandatory_risk_reasons():
    rows = [
        _candidate(1, 0.000, 0.250, -1.000e-3),
        _candidate(2, 0.600, 0.002, -0.800e-3),
        _candidate(3, 0.604, 0.002, -0.799e-3, edge_risk=True, low_energy=True),
        _candidate(4, -0.500, 0.400, -0.200e-3),
    ]

    clustered = cluster_branch_candidates(
        rows,
        coarse_dq=0.004,
        coarse_dDelta=0.002,
        numerical_energy_scale=1.0e-5,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
    )
    duplicate = next(row for row in clustered if row["merged_branch_ids"] == "2;3")

    assert duplicate["edge_risk"] is True
    assert duplicate["within_low_energy_window"] is True
    assert duplicate["mandatory_basin"] is True
    assert duplicate["basin_has_delta_near_epsilon"] is True
    assert "Delta_near_epsilon" in str(duplicate["basin_risk_flags"])
    reasons = set(str(duplicate["mandatory_basin_reasons"]).split(";"))
    assert {"Delta_near_epsilon", "edge_risk", "near_degenerate"} <= reasons
    assert "edge_risk" in _local_refine_selection_reason(duplicate, delta_eps=1.0e-3, delta_refine_half_width=0.003)


def test_basin_clustering_keeps_separated_mandatory_basins_distinct():
    rows = [
        _candidate(1, 0.000, 0.250, -1.000e-3),
        _candidate(2, -0.900, 0.250, -0.990e-3, edge_risk=True),
        _candidate(3, 0.900, 0.250, -0.989e-3, edge_risk=True),
    ]

    clustered = cluster_branch_candidates(
        rows,
        coarse_dq=0.004,
        coarse_dDelta=0.002,
        numerical_energy_scale=1.0e-5,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.03,
    )

    assert len(clustered) == 3
    edge_basins = [row for row in clustered if row["edge_risk"]]
    assert len(edge_basins) == 2
    assert {row["merged_branch_ids"] for row in edge_basins} == {"2", "3"}


def test_basin_level_risk_from_nonrepresentative_member_controls_pruning_and_selection():
    rows = [
        _candidate(1, 0.000, 0.250, -1.0000e-3),
        _candidate(2, 0.500, 0.200, -0.9000e-3),
        _candidate(3, 0.501, 0.002, -0.8999e-3),
    ]

    clustered = cluster_branch_candidates(
        rows,
        coarse_dq=0.004,
        coarse_dDelta=0.25,
        numerical_energy_scale=1.0e-3,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
    )
    merged = next(row for row in clustered if row["merged_branch_ids"] == "2;3")

    assert int(merged["minimum_rank"]) == 2
    assert merged["basin_has_delta_near_epsilon"] is True
    assert merged["mandatory_basin"] is True
    assert "Delta_near_epsilon" in str(merged["mandatory_basin_reasons"])

    marked = mark_energy_window_pruning(
        clustered,
        local_refine_energy_window=1.0e-6,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        enabled=True,
    )
    marked_merged = next(row for row in marked if row["merged_branch_ids"] == "2;3")
    assert marked_merged["pruned_by_energy_window"] is False

    selected = select_local_refine_targets(
        marked,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        max_total_refined_basins=1,
        enable_selective_refinement=True,
        max_optional_refined_basins=0,
        mandatory_basins_can_exceed_cap=True,
    )

    selected_ids = {int(row["minimum_rank"]) for row in selected}
    assert selected_ids == {1, 2}
