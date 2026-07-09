from __future__ import annotations

from ml_phase.exact_oracle import (
    ADAPTIVE_LOCAL_BOX_DIAGNOSTIC_FIELDS,
    adaptive_local_box_half_widths,
    build_adaptive_local_box_diagnostic_record,
    estimate_basin_geometry,
)


def test_estimate_basin_geometry_records_widths_and_curvature_proxy():
    rows = [
        {"q_local_min": 0.10, "Delta_local_min": 0.20, "DeltaF_local_min": -1.00e-3},
        {"q_local_min": 0.14, "Delta_local_min": 0.23, "DeltaF_local_min": -0.99e-3},
    ]

    geometry = estimate_basin_geometry(rows)

    assert round(geometry["basin_q_width"], 12) == 0.04
    assert round(geometry["basin_Delta_width"], 12) == 0.03
    assert round(geometry["basin_energy_span"], 12) == 1.0e-5
    assert geometry["basin_curvature_proxy"] > 0.0


def test_adaptive_local_box_defaults_to_fixed_box_when_disabled():
    row = {"basin_q_width": 0.20, "basin_Delta_width": 0.10}

    q_half, delta_half = adaptive_local_box_half_widths(
        row,
        default_q_half_width=0.03,
        default_delta_half_width=0.01,
        enabled=False,
    )

    assert q_half == 0.03
    assert delta_half == 0.01


def test_adaptive_local_box_suggestion_is_bounded_when_enabled():
    row = {"basin_q_width": 0.20, "basin_Delta_width": 0.10}

    q_half, delta_half = adaptive_local_box_half_widths(
        row,
        default_q_half_width=0.03,
        default_delta_half_width=0.01,
        enabled=True,
        min_factor=0.5,
        max_factor=2.0,
    )

    assert q_half == 0.06
    assert delta_half == 0.02


def test_adaptive_local_box_diagnostic_record_keeps_fixed_box_when_disabled():
    row = {
        "basin_q_width": 0.20,
        "basin_Delta_width": 0.10,
        "basin_energy_span": 1.0e-5,
        "basin_curvature_proxy": 0.01,
    }

    record = build_adaptive_local_box_diagnostic_record(
        row,
        default_q_half_width=0.03,
        default_delta_half_width=0.01,
        enabled=False,
    )

    assert list(record.keys()) == ADAPTIVE_LOCAL_BOX_DIAGNOSTIC_FIELDS
    assert record["adaptive_box_enabled"] == 0
    assert record["suggested_q_half_width"] == 0.03
    assert record["suggested_delta_half_width"] == 0.01
    assert record["adaptive_box_reason"] == "adaptive_disabled_fixed_box"
    assert record["basin_q_width"] == 0.20
    assert record["basin_Delta_width"] == 0.10


def test_adaptive_local_box_diagnostic_record_names_bounded_suggestion():
    row = {
        "basin_q_width": 0.20,
        "basin_Delta_width": 0.10,
        "basin_energy_span": 1.0e-5,
        "basin_curvature_proxy": 0.01,
    }

    record = build_adaptive_local_box_diagnostic_record(
        row,
        default_q_half_width=0.03,
        default_delta_half_width=0.01,
        enabled=True,
        min_factor=0.5,
        max_factor=2.0,
    )

    assert record["adaptive_box_enabled"] == 1
    assert record["suggested_q_half_width"] == 0.06
    assert record["suggested_delta_half_width"] == 0.02
    assert record["adaptive_box_reason"] == "bounded_by_max_factor"
