import numpy as np

from ml_phase.exact_oracle import QScanCache, _merge_q_scan_caches


def test_qscan_cache_merge_sorts_and_deduplicates():
    left = QScanCache(
        q_values=np.array([-2.0, -1.0]),
        delta_star_q=np.array([0.2, 0.3]),
        deltaf_min_q=np.array([-0.2, -0.3]),
        omega_min_q=np.array([1.0, 0.9]),
        omega_normal_scalar=1.2,
        source_level=np.array([1, 1]),
    )
    right = QScanCache(
        q_values=np.array([-1.0, 0.0]),
        delta_star_q=np.array([0.4, 0.5]),
        deltaf_min_q=np.array([-0.35, -0.1]),
        omega_min_q=np.array([0.85, 1.1]),
        omega_normal_scalar=1.2,
        source_level=np.array([2, 2]),
    )

    merged = _merge_q_scan_caches([right, left])

    assert np.allclose(merged.q_values, [-2.0, -1.0, 0.0])
    assert np.allclose(merged.deltaf_min_q, [-0.2, -0.35, -0.1])
    assert np.allclose(merged.delta_star_q, [0.2, 0.4, 0.5])
