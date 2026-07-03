import numpy as np

from ml_phase.exact_oracle import QScanCache, _cache_to_scan, _merge_q_scan_caches


def test_incremental_cache_global_minimum_can_move_to_added_strip():
    base = QScanCache(
        q_values=np.array([-1.0, 0.0, 1.0]),
        delta_star_q=np.array([0.1, 0.2, 0.1]),
        deltaf_min_q=np.array([-0.01, -0.02, -0.01]),
        omega_min_q=np.array([0.99, 0.98, 0.99]),
        omega_normal_scalar=1.0,
        source_level=np.array([0, 0, 0]),
    )
    strip = QScanCache(
        q_values=np.array([2.0]),
        delta_star_q=np.array([0.4]),
        deltaf_min_q=np.array([-0.05]),
        omega_min_q=np.array([0.95]),
        omega_normal_scalar=1.0,
        source_level=np.array([1]),
    )

    scan = _cache_to_scan(_merge_q_scan_caches([base, strip]), q_edge_margin=None)

    assert np.isclose(scan.q_opt, 2.0)
    assert np.isclose(scan.delta_opt, 0.4)
    assert np.isclose(scan.deltaf_min, -0.05)
