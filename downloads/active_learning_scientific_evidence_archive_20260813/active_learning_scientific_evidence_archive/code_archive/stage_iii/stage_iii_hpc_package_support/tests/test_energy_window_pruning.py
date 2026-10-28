from __future__ import annotations

from ml_phase.exact_oracle import mark_energy_window_pruning, select_local_refine_targets


def _candidate(
    rank: int,
    energy_above: float,
    *,
    delta: float = 0.2,
    edge_risk: bool = False,
    low_energy: bool = False,
) -> dict[str, object]:
    return {
        "minimum_rank": rank,
        "grid_index": rank,
        "q_local_min": 0.1 * rank,
        "Delta_local_min": delta,
        "DeltaF_local_min": -1.0e-3 + energy_above,
        "energy_above_global": energy_above,
        "distance_to_q_edge": 0.01 if edge_risk else 0.2,
        "within_low_energy_window": low_energy,
        "edge_risk": edge_risk,
    }


def test_energy_window_pruning_marks_only_ordinary_high_energy_basins():
    rows = [
        _candidate(1, 0.0),
        _candidate(2, 5.0e-4),
        _candidate(3, 6.0e-4, edge_risk=True),
        _candidate(4, 7.0e-4, delta=0.002),
        _candidate(5, 8.0e-4, low_energy=True),
    ]

    marked = mark_energy_window_pruning(
        rows,
        local_refine_energy_window=1.0e-4,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        enabled=True,
    )

    pruned = {int(row["minimum_rank"]) for row in marked if row["pruned_by_energy_window"]}
    kept = {int(row["minimum_rank"]) for row in marked if not row["pruned_by_energy_window"]}
    assert pruned == {2}
    assert kept == {1, 3, 4, 5}
    assert next(row for row in marked if int(row["minimum_rank"]) == 2)["pruned_reason"] == "ordinary_above_energy_window"


def test_energy_window_pruning_is_default_off():
    rows = [_candidate(1, 0.0), _candidate(2, 5.0e-4)]

    marked = mark_energy_window_pruning(
        rows,
        local_refine_energy_window=1.0e-4,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        enabled=False,
    )

    assert not any(row["pruned_by_energy_window"] for row in marked)


def test_pruned_ordinary_basins_are_skipped_by_refinement_selection():
    rows = [
        _candidate(1, 0.0),
        _candidate(2, 5.0e-4),
        _candidate(3, 6.0e-4, edge_risk=True),
        _candidate(4, 7.0e-4),
    ]
    marked = mark_energy_window_pruning(
        rows,
        local_refine_energy_window=1.0e-4,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        enabled=True,
    )

    selected = select_local_refine_targets(
        marked,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        max_total_refined_basins=6,
        enable_selective_refinement=True,
        max_optional_refined_basins=3,
        mandatory_basins_can_exceed_cap=True,
    )

    assert [int(row["minimum_rank"]) for row in selected] == [1, 3]


def test_rank_and_cap_energy_window_prunes_only_ordinary_basins():
    rows = [
        _candidate(1, 0.0),
        _candidate(2, 5.0e-4, edge_risk=True),
        _candidate(3, 6.0e-4, delta=0.002),
        _candidate(4, 7.0e-4, low_energy=True),
        _candidate(5, 5.0e-5),
        _candidate(6, 5.0e-4),
        _candidate(7, 6.0e-4),
    ]
    marked = mark_energy_window_pruning(
        rows,
        local_refine_energy_window=1.0e-4,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        enabled=True,
    )

    selected = select_local_refine_targets(
        marked,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        max_total_refined_basins=6,
        enable_selective_refinement=True,
        max_optional_refined_basins=3,
        mandatory_basins_can_exceed_cap=False,
        high_risk_overflow_policy="rank_and_cap",
        max_edge_risk_basins=1,
        max_delta_near_eps_basins=1,
        max_near_degenerate_basins=1,
    )

    pruned = {int(row["minimum_rank"]) for row in marked if row["pruned_by_energy_window"]}
    assert pruned == {6, 7}
    assert {int(row["minimum_rank"]) for row in selected} == {1, 2, 3, 4, 5}
    assert all(int(row["minimum_rank"]) not in pruned for row in selected)
