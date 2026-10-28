from dataclasses import replace

import numpy as np

from eta_phase_diagram_cuda import EtaPhaseConfig
from ml_phase.exact_oracle import ScanResult, _incremental_q_strips


def _scan(q):
    return ScanResult(
        q_vec=np.asarray(q, dtype=float),
        delta_star_q=np.zeros(len(q)),
        deltaf_q=np.zeros(len(q)),
        omega_sc_q=np.zeros(len(q)),
        omega_normal_q=np.zeros(len(q)),
        omega_normal_scalar=0.0,
        q_opt=0.0,
        delta_opt=0.0,
        deltaf_min=0.0,
        q_index=0,
        dq=1.0,
        q_min=float(q[0]),
        q_max=float(q[-1]),
        n_q=len(q),
        q_edge_margin=1.0,
        qopt_edge_hit=0,
        q_edge_distance=1.0,
    )


def test_incremental_q_strips_preserve_old_spacing():
    old = _scan([-1.0, 0.0, 1.0])
    cfg = replace(EtaPhaseConfig(), q_min=-3.0, q_max=2.0, n_q=6)

    left, right = _incremental_q_strips(old, cfg)

    assert np.allclose(left, [-3.0, -2.0])
    assert np.allclose(right, [2.0])
