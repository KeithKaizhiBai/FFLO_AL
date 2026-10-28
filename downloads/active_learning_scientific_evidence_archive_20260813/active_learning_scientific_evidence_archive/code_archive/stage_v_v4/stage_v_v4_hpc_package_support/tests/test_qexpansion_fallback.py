from ml_phase.exact_oracle import ConfirmedPoint, _rows_to_result


def test_fallback_full_rescan_reason_is_saved_as_string():
    row = ConfirmedPoint(
        kT=0.0,
        JA=0.0,
        eta=0.0,
        q_opt=0.0,
        delta_opt=0.0,
        ic_plus=0.0,
        ic_minus=0.0,
        phase_candidate=0,
        q_status=0,
        q_min=-1.0,
        q_max=1.0,
        n_q=3,
        q_index=1,
        q_edge_distance=1.0,
        q_edge_hit=0,
        q_refinement_level=0,
        q_expanded=0,
        q_unresolved=0,
        delta_status=0,
        delta_min=0.0,
        delta_max=1.0,
        n_delta=3,
        n_delta_refined=0,
        delta_refinement_level=0,
        delta_boundary_ambiguous=0,
        delta_refined=0,
        delta_unresolved=0,
        free_energy_gap_to_normal=1.0,
        positive_delta_gap=1.0,
        positive_delta_checked=1,
        exact_status_code=0,
        exact_status_name="trusted",
        trusted_exact=1,
        fallback_full_rescan_used=1,
        fallback_full_rescan_reason="no_incremental_strip",
    )

    result = _rows_to_result([row])

    assert result.fallback_full_rescan_used[0] == 1
    assert result.fallback_full_rescan_reason[0] == "no_incremental_strip"
