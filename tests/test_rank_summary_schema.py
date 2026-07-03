REQUIRED_RANK_SUMMARY_FIELDS = {
    "rank",
    "world_size",
    "n_points",
    "elapsed_sec",
    "hostname",
    "device",
    "oracle_mode",
    "enable_incremental_q_expansion",
    "point_total_runtime_sec_sum",
    "base_scan_runtime_sec_sum",
    "q_expansion_runtime_sec_sum",
    "delta_refinement_runtime_sec_sum",
    "local_refinement_runtime_sec_sum",
    "total_q_points_evaluated",
    "total_estimated_grid_evaluations",
    "incremental_expansion_used_count",
    "fallback_full_rescan_used_count",
}


def test_rank_summary_schema_contains_timing_and_workload_fields():
    expected = {
        "rank",
        "world_size",
        "n_points",
        "elapsed_sec",
        "hostname",
        "device",
        "oracle_mode",
        "enable_incremental_q_expansion",
        "point_total_runtime_sec_sum",
        "base_scan_runtime_sec_sum",
        "q_expansion_runtime_sec_sum",
        "delta_refinement_runtime_sec_sum",
        "local_refinement_runtime_sec_sum",
        "total_q_points_evaluated",
        "total_estimated_grid_evaluations",
        "incremental_expansion_used_count",
        "fallback_full_rescan_used_count",
    }

    assert expected == REQUIRED_RANK_SUMMARY_FIELDS
