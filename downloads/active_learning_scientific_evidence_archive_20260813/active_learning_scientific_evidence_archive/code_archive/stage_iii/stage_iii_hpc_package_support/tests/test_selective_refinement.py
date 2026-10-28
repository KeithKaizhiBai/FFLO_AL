from __future__ import annotations

from ml_phase.exact_oracle import select_local_refine_targets


def _candidate(
    rank: int,
    deltaf: float,
    *,
    delta: float = 0.2,
    edge_risk: bool = False,
    low_energy: bool = False,
) -> dict[str, object]:
    return {
        "minimum_rank": rank,
        "grid_index": rank,
        "q_local_min": 0.05 * rank,
        "Delta_local_min": delta,
        "DeltaF_local_min": deltaf,
        "energy_above_global": deltaf + 1.0e-3,
        "distance_to_q_edge": 0.01 if edge_risk else 0.2,
        "within_low_energy_window": low_energy,
        "edge_risk": edge_risk,
    }


def test_legacy_refine_target_selection_preserves_existing_cap_behavior():
    rows = [
        _candidate(1, -1.00e-3),
        _candidate(2, -0.99e-3, edge_risk=True),
        _candidate(3, -0.98e-3, edge_risk=True),
        _candidate(4, -0.97e-3, low_energy=True),
        _candidate(5, -0.96e-3, delta=0.002),
    ]

    selected = select_local_refine_targets(
        rows,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        max_total_refined_basins=3,
        enable_selective_refinement=False,
    )

    assert [int(row["minimum_rank"]) for row in selected] == [1, 2, 3]


def test_selective_refinement_keeps_all_mandatory_basins_even_above_total_cap():
    rows = [
        _candidate(1, -1.00e-3),
        _candidate(2, -0.99e-3, edge_risk=True),
        _candidate(3, -0.98e-3, edge_risk=True),
        _candidate(4, -0.97e-3, low_energy=True),
        _candidate(5, -0.96e-3, delta=0.002),
        _candidate(6, -0.95e-3),
        _candidate(7, -0.94e-3),
    ]

    selected = select_local_refine_targets(
        rows,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        max_total_refined_basins=3,
        enable_selective_refinement=True,
        max_optional_refined_basins=1,
        mandatory_basins_can_exceed_cap=True,
    )

    selected_ids = [int(row["minimum_rank"]) for row in selected]
    assert selected_ids == [1, 2, 3, 4, 5, 6]
    assert all(row["mandatory_basin"] for row in selected if int(row["minimum_rank"]) <= 5)
    assert selected[-1]["mandatory_basin"] is False


def test_selective_refinement_can_enforce_total_cap_when_configured():
    rows = [
        _candidate(1, -1.00e-3),
        _candidate(2, -0.99e-3, edge_risk=True),
        _candidate(3, -0.98e-3, edge_risk=True),
        _candidate(4, -0.97e-3, low_energy=True),
        _candidate(5, -0.96e-3, delta=0.002),
    ]

    selected = select_local_refine_targets(
        rows,
        delta_eps=1.0e-3,
        delta_refine_half_width=0.003,
        max_total_refined_basins=3,
        enable_selective_refinement=True,
        max_optional_refined_basins=2,
        mandatory_basins_can_exceed_cap=False,
    )

    assert [int(row["minimum_rank"]) for row in selected] == [1, 2, 3]


def test_rank_and_cap_limits_mandatory_risks_and_enforces_total_cap():
    rows = [
        _candidate(1, -1.00e-3),
        _candidate(2, -0.99e-3, edge_risk=True),
        _candidate(3, -0.98e-3, edge_risk=True),
        _candidate(4, -0.97e-3, delta=0.0012),
        _candidate(5, -0.96e-3, delta=0.0014),
        _candidate(6, -0.95e-3, low_energy=True),
        _candidate(7, -0.94e-3, low_energy=True),
        _candidate(8, -0.93e-3),
    ]

    selected = select_local_refine_targets(
        rows,
        delta_eps=1.0e-3,
        delta_refine_half_width=5.0e-4,
        max_total_refined_basins=4,
        enable_selective_refinement=True,
        max_optional_refined_basins=2,
        mandatory_basins_can_exceed_cap=False,
        high_risk_overflow_policy="rank_and_cap",
        max_edge_risk_basins=1,
        max_delta_near_eps_basins=1,
        max_near_degenerate_basins=1,
    )

    assert [int(row["minimum_rank"]) for row in selected] == [1, 2, 4, 6]
    assert all(row["mandatory_basin"] for row in selected)
    assert all(row["mandatory_overflow_policy"] == "rank_and_cap" for row in selected)
    assert all(row["mandatory_overflow"] is True for row in selected)


def test_rank_and_cap_fills_remaining_slots_with_ordinary_optional_basins():
    rows = [
        _candidate(1, -1.00e-3),
        _candidate(2, -0.99e-3, edge_risk=True),
        _candidate(3, -0.98e-3, edge_risk=True),
        _candidate(4, -0.97e-3, delta=0.0012),
        _candidate(5, -0.96e-3, delta=0.0014),
        _candidate(6, -0.95e-3, low_energy=True),
        _candidate(7, -0.94e-3, low_energy=True),
        _candidate(8, -0.93e-3),
        _candidate(9, -0.92e-3),
        _candidate(10, -0.91e-3),
    ]

    selected = select_local_refine_targets(
        rows,
        delta_eps=1.0e-3,
        delta_refine_half_width=5.0e-4,
        max_total_refined_basins=6,
        enable_selective_refinement=True,
        max_optional_refined_basins=2,
        mandatory_basins_can_exceed_cap=False,
        high_risk_overflow_policy="rank_and_cap",
        max_edge_risk_basins=1,
        max_delta_near_eps_basins=1,
        max_near_degenerate_basins=1,
    )

    assert [int(row["minimum_rank"]) for row in selected] == [1, 2, 4, 6, 8, 9]
    optional = [row for row in selected if not row["mandatory_basin"]]
    assert [int(row["minimum_rank"]) for row in optional] == [8, 9]
