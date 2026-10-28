from __future__ import annotations

from ml_phase.exact_oracle import (
    BRANCH_REUSE_CACHE_FIELDS,
    BRANCH_REUSE_DIAGNOSTIC_FIELDS,
    branch_reuse_signature,
    build_branch_reuse_cache_record,
    build_branch_reuse_diagnostic_record,
    evaluate_branch_reuse_candidate,
)


def _candidate() -> dict[str, object]:
    return {
        "q_local_min": 0.12,
        "Delta_local_min": 0.20,
        "DeltaF_local_min": -1.0e-3,
    }


def _cached(solver_sig: str, box_sig: str) -> dict[str, object]:
    return {
        "q_refined": 0.121,
        "Delta_refined": 0.201,
        "DeltaF_refined": -1.0005e-3,
        "solver_config_signature": solver_sig,
        "local_box_signature": box_sig,
    }


def test_branch_reuse_signature_is_stable_for_key_order():
    left = branch_reuse_signature({"n_q": 800, "q_min": -0.1, "q_max": 0.1})
    right = branch_reuse_signature({"q_max": 0.1, "q_min": -0.1, "n_q": 800})

    assert left == right


def test_branch_reuse_accepts_matching_branch_and_signatures():
    solver_sig = branch_reuse_signature({"solver": "robust_incremental"})
    box_sig = branch_reuse_signature({"n_q_local": 800, "n_delta_local": 600})

    decision = evaluate_branch_reuse_candidate(
        _cached(solver_sig, box_sig),
        _candidate(),
        solver_config_signature=solver_sig,
        local_box_signature=box_sig,
        q_tolerance=0.005,
        delta_tolerance=0.005,
        energy_tolerance=1.0e-5,
    )

    assert decision["reuse_allowed"] is True
    assert decision["reuse_rejection_reason"] == "reuse_allowed"
    assert decision["refined_q"] == 0.121


def test_branch_reuse_rejects_signature_mismatch():
    solver_sig = branch_reuse_signature({"solver": "robust_incremental"})
    box_sig = branch_reuse_signature({"n_q_local": 800})

    decision = evaluate_branch_reuse_candidate(
        _cached("wrong", box_sig),
        _candidate(),
        solver_config_signature=solver_sig,
        local_box_signature=box_sig,
        q_tolerance=0.005,
        delta_tolerance=0.005,
        energy_tolerance=1.0e-5,
    )

    assert decision == {
        "reuse_allowed": False,
        "reuse_rejection_reason": "solver_config_signature_mismatch",
    }


def test_branch_reuse_rejects_lower_energy_competing_branch():
    solver_sig = branch_reuse_signature({"solver": "robust_incremental"})
    box_sig = branch_reuse_signature({"n_q_local": 800})

    decision = evaluate_branch_reuse_candidate(
        _cached(solver_sig, box_sig),
        _candidate(),
        solver_config_signature=solver_sig,
        local_box_signature=box_sig,
        q_tolerance=0.005,
        delta_tolerance=0.005,
        energy_tolerance=1.0e-5,
        lower_competing_deltaf=-1.1e-3,
    )

    assert decision == {
        "reuse_allowed": False,
        "reuse_rejection_reason": "lower_energy_competing_branch",
    }


def test_branch_reuse_cache_record_has_explicit_integration_fields():
    solver_sig = branch_reuse_signature({"solver": "robust_incremental"})
    box_sig = branch_reuse_signature({"n_q_local": 800})
    candidate = {**_candidate(), "minimum_rank": 2, "basin_id": 7}
    refined = {"refined_q": 0.122, "refined_Delta": 0.203, "refined_DeltaF": -1.001e-3}

    record = build_branch_reuse_cache_record(
        refined,
        candidate,
        solver_config_signature=solver_sig,
        local_box_signature=box_sig,
        point_id=11,
        q_window_level=3,
    )

    assert list(record.keys()) == BRANCH_REUSE_CACHE_FIELDS
    assert record["point_id"] == 11
    assert record["branch_id"] == 2
    assert record["basin_id"] == 7
    assert record["cluster_id"] == 7
    assert record["q_window_level"] == 3
    assert record["reuse_cache_valid"] == 1
    assert record["solver_config_signature"] == solver_sig
    assert record["local_box_signature"] == box_sig


def test_branch_reuse_diagnostic_record_is_explicit_for_allowed_reuse():
    solver_sig = branch_reuse_signature({"solver": "robust_incremental"})
    box_sig = branch_reuse_signature({"n_q_local": 800, "n_delta_local": 600})
    candidate = {**_candidate(), "minimum_rank": 4, "basin_id": 5}
    cached = _cached(solver_sig, box_sig)
    decision = evaluate_branch_reuse_candidate(
        cached,
        candidate,
        solver_config_signature=solver_sig,
        local_box_signature=box_sig,
        q_tolerance=0.005,
        delta_tolerance=0.005,
        energy_tolerance=1.0e-5,
    )

    record = build_branch_reuse_diagnostic_record(
        cached,
        candidate,
        decision,
        solver_config_signature=solver_sig,
        local_box_signature=box_sig,
    )

    assert list(record.keys()) == BRANCH_REUSE_DIAGNOSTIC_FIELDS
    assert record["branch_id"] == 4
    assert record["basin_id"] == 5
    assert record["reuse_attempted"] == 1
    assert record["reuse_allowed"] == 1
    assert record["reuse_rejection_reason"] == "reuse_allowed"
    assert record["q_abs_diff"] > 0.0
    assert record["Delta_abs_diff"] > 0.0
    assert record["energy_abs_diff"] > 0.0


def test_branch_reuse_diagnostic_record_is_explicit_for_missing_cache():
    solver_sig = branch_reuse_signature({"solver": "robust_incremental"})
    box_sig = branch_reuse_signature({"n_q_local": 800})
    candidate = {**_candidate(), "minimum_rank": 1}
    decision = evaluate_branch_reuse_candidate(
        None,
        candidate,
        solver_config_signature=solver_sig,
        local_box_signature=box_sig,
        q_tolerance=0.005,
        delta_tolerance=0.005,
        energy_tolerance=1.0e-5,
    )

    record = build_branch_reuse_diagnostic_record(
        None,
        candidate,
        decision,
        solver_config_signature=solver_sig,
        local_box_signature=box_sig,
    )

    assert record["reuse_attempted"] == 0
    assert record["reuse_allowed"] == 0
    assert record["reuse_rejection_reason"] == "missing_cached_branch"
    assert record["cached_solver_config_signature"] == ""
    assert record["cached_local_box_signature"] == ""
