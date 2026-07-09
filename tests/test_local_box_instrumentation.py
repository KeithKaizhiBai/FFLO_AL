from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_phase import exact_oracle
from ml_phase.exact_oracle import ConfirmedPoint, LOCAL_BOX_RECORD_FIELDS, evaluate_points


def _fake_confirmed_point(kT: float, JA: float) -> ConfirmedPoint:
    return ConfirmedPoint(
        kT=float(kT),
        JA=float(JA),
        eta=0.0,
        q_opt=0.0,
        delta_opt=0.2,
        ic_plus=0.0,
        ic_minus=0.0,
        phase_candidate=1,
        q_status=1,
        q_min=-0.5,
        q_max=0.5,
        n_q=10,
        q_index=5,
        q_edge_distance=0.5,
        q_edge_hit=0,
        q_refinement_level=0,
        q_expanded=0,
        q_unresolved=0,
        delta_status=0,
        delta_min=0.0,
        delta_max=1.0,
        n_delta=10,
        n_delta_refined=0,
        delta_refinement_level=0,
        delta_boundary_ambiguous=0,
        delta_refined=0,
        delta_unresolved=0,
        free_energy_gap_to_normal=-1e-3,
        positive_delta_gap=float("nan"),
        positive_delta_checked=0,
        exact_status_code=0,
        exact_status_name="trusted",
        trusted_exact=1,
    )


def test_local_box_instrumentation_writes_schema_when_enabled(tmp_path, monkeypatch):
    def fake_confirm_one_point(*args, **kwargs):
        records = kwargs.get("local_box_records")
        if records is not None:
            records.append({field: 0 for field in LOCAL_BOX_RECORD_FIELDS})
            records[-1].update(
                {
                    "point_id": kwargs.get("point_index", 0),
                    "kT": args[0],
                    "JA": args[1],
                    "selection_reason": "global_best",
                    "refined_status": "refined_box",
                    "pruned_reason": "not_pruned_stage_1",
                }
            )
        return _fake_confirmed_point(args[0], args[1])

    monkeypatch.setattr(exact_oracle, "_confirm_one_point", fake_confirm_one_point)
    local_box_csv = tmp_path / "performance" / "local_box.csv"
    result = evaluate_points(
        points=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        output_file=None,
        save_every=0,
        enable_local_box_instrumentation=True,
        local_box_output_file=local_box_csv,
    )

    assert result.kT.shape[0] == 2
    assert local_box_csv.exists()
    table = pd.read_csv(local_box_csv)
    assert list(table.columns) == LOCAL_BOX_RECORD_FIELDS
    assert table.shape[0] == 2
    assert set(table["pruned_reason"]) == {"not_pruned_stage_1"}


def test_local_box_instrumentation_default_off_writes_no_file(tmp_path, monkeypatch):
    def fake_confirm_one_point(*args, **kwargs):
        assert kwargs.get("local_box_records") is None
        return _fake_confirmed_point(args[0], args[1])

    monkeypatch.setattr(exact_oracle, "_confirm_one_point", fake_confirm_one_point)
    local_box_csv = tmp_path / "performance" / "local_box.csv"
    evaluate_points(
        points=np.array([[0.1, 0.2]], dtype=float),
        output_file=None,
        save_every=0,
        enable_local_box_instrumentation=False,
        local_box_output_file=local_box_csv,
    )

    assert not local_box_csv.exists()
