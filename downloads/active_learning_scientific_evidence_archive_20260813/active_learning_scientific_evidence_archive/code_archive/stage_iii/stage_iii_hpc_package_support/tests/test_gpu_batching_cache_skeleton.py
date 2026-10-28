from __future__ import annotations

from ml_phase.exact_oracle import (
    HAMILTONIAN_CACHE_DIAGNOSTIC_FIELDS,
    LOCAL_BOX_BATCH_PLAN_FIELDS,
    LOCAL_BOX_PROFILER_EVENT_FIELDS,
    build_hamiltonian_cache_diagnostic_record,
    build_hamiltonian_cache_signature,
    build_local_box_batch_plan,
    build_local_box_profiler_event,
    evaluate_hamiltonian_cache_candidate,
)


def _box(branch_id: int, q_lo: float, q_hi: float) -> dict[str, object]:
    return {
        "point_id": 12,
        "kT": 0.04,
        "JA": 1.25,
        "branch_id": branch_id,
        "q_window_level": 2,
        "box_q_min": q_lo,
        "box_q_max": q_hi,
        "box_Delta_min": 0.01,
        "box_Delta_max": 0.05,
        "n_q_local": 41,
        "n_Delta_local": 31,
        "local_grid_evaluations": 1271,
    }


def test_local_box_batch_plan_defaults_to_single_box_records_when_disabled():
    plan = build_local_box_batch_plan(
        [_box(1, -0.04, 0.04), _box(2, 0.02, 0.10)],
        enabled=False,
        max_boxes_per_batch=8,
        dtype="float64",
        device="cuda",
    )

    assert len(plan) == 2
    assert list(plan[0].keys()) == LOCAL_BOX_BATCH_PLAN_FIELDS
    assert plan[0]["batching_enabled"] == 0
    assert plan[0]["box_count"] == 1
    assert plan[0]["branch_ids"] == "1"
    assert plan[0]["grid_shape"] == "boxes=1;n_q_max=41;n_Delta_max=31"
    assert plan[0]["batch_plan_reason"] == "batching_disabled_single_box"


def test_local_box_batch_plan_groups_boxes_when_enabled():
    plan = build_local_box_batch_plan(
        [_box(1, -0.04, 0.04), _box(2, 0.02, 0.10), _box(3, 0.12, 0.20)],
        enabled=True,
        max_boxes_per_batch=2,
    )

    assert len(plan) == 2
    assert plan[0]["batching_enabled"] == 1
    assert plan[0]["box_count"] == 2
    assert plan[0]["branch_ids"] == "1;2"
    assert plan[0]["local_grid_evaluations"] == 2542
    assert plan[0]["box_q_min"] == -0.04
    assert plan[0]["box_q_max"] == 0.10
    assert plan[0]["batch_plan_reason"] == "enabled_chunked_batch"
    assert plan[1]["box_count"] == 1


def test_hamiltonian_cache_signature_is_stable_for_key_order_and_ignores_extras():
    left = build_hamiltonian_cache_signature(
        {
            "solver_mode": "robust_incremental",
            "dtype": "float64",
            "device": "cuda",
            "n_k": 800,
            "kT": 0.04,
            "JA": 1.25,
            "extra_debug_field": "ignored",
        }
    )
    right = build_hamiltonian_cache_signature(
        {
            "JA": 1.25,
            "kT": 0.04,
            "n_k": 800,
            "device": "cuda",
            "dtype": "float64",
            "solver_mode": "robust_incremental",
        }
    )

    assert left == right


def test_hamiltonian_cache_candidate_requires_matching_signature_shape_dtype_and_device():
    signature = build_hamiltonian_cache_signature(
        {
            "solver_mode": "robust_incremental",
            "dtype": "float64",
            "device": "cuda",
            "n_k": 800,
            "kT": 0.04,
            "JA": 1.25,
            "q_grid_signature": "q-local",
            "delta_grid_signature": "delta-local",
            "local_box_signature": "box-1",
        }
    )
    cached = {
        "hamiltonian_cache_signature": signature,
        "tensor_shape": (1, 41, 31, 800),
        "dtype": "float64",
        "device": "cuda",
    }

    decision = evaluate_hamiltonian_cache_candidate(
        cached,
        expected_cache_signature=signature,
        expected_tensor_shape=(1, 41, 31, 800),
        expected_dtype="float64",
        expected_device="cuda",
        cache_enabled=True,
    )

    assert decision == {
        "cache_hit_allowed": True,
        "cache_rejection_reason": "cache_hit_allowed",
    }

    rejected = evaluate_hamiltonian_cache_candidate(
        {**cached, "tensor_shape": (2, 41, 31, 800)},
        expected_cache_signature=signature,
        expected_tensor_shape=(1, 41, 31, 800),
        expected_dtype="float64",
        expected_device="cuda",
        cache_enabled=True,
    )

    assert rejected == {
        "cache_hit_allowed": False,
        "cache_rejection_reason": "tensor_shape_mismatch",
    }


def test_hamiltonian_cache_diagnostic_record_is_explicit_for_missing_cache():
    decision = evaluate_hamiltonian_cache_candidate(
        None,
        expected_cache_signature="expected",
        expected_tensor_shape=(1, 41, 31, 800),
        expected_dtype="float64",
        expected_device="cuda",
        cache_enabled=True,
    )

    record = build_hamiltonian_cache_diagnostic_record(
        None,
        decision,
        expected_cache_signature="expected",
        expected_tensor_shape=(1, 41, 31, 800),
        expected_dtype="float64",
        expected_device="cuda",
        cache_enabled=True,
    )

    assert list(record.keys()) == HAMILTONIAN_CACHE_DIAGNOSTIC_FIELDS
    assert record["cache_enabled"] == 1
    assert record["cache_lookup_attempted"] == 1
    assert record["cache_hit_allowed"] == 0
    assert record["cache_rejection_reason"] == "missing_cached_entry"
    assert record["expected_tensor_shape"] == "1x41x31x800"
    assert record["cached_tensor_shape"] == ""


def test_local_box_profiler_event_has_stable_fields():
    event = build_local_box_profiler_event(
        profiler_scope="local_box_scan",
        event_name="tensor_constructed",
        point_id=12,
        branch_id=2,
        batch_id=0,
        batching_enabled=True,
        cache_lookup_attempted=True,
        cache_hit_allowed=False,
        tensor_construction_location="_run_scan_for_q_vec_with_normal",
        grid_shape=(2, 41, 31, 800),
        runtime_sec=0.25,
        local_grid_evaluations=2542,
        profiler_note="fixture",
    )

    assert list(event.keys()) == LOCAL_BOX_PROFILER_EVENT_FIELDS
    assert event["batching_enabled"] == 1
    assert event["cache_lookup_attempted"] == 1
    assert event["cache_hit_allowed"] == 0
    assert event["grid_shape"] == "2x41x31x800"
    assert event["runtime_sec"] == 0.25
