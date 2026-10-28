# Stage III evidence extract

## Frozen thermodynamic/topology hierarchy

The thermodynamic phase labels remain normal, uniform-SC, and FFLO. For
superconducting points, the topology pass separately evaluates (P_0),
(P_\pi), and the full-Brillouin-zone bulk gap. A Pfaffian sign is used as a
topology label only for a trusted gapped state. Gapless and unresolved states
remain distinct.

Source: `E-MODEL-SPEC`, `E-NUMERICS-SPEC`, `E-STAGEIII-PASS`.

## Oracle validation

The Stage III offline pass reports analytic/numerical Pfaffian-product sign
agreement for 104/104 non-boundary cases and a maximum CPU/GPU bulk-gap
difference of (7.77\times10^{-16}) at (N_k=2048). These numbers validate
the implementation/convention; they do not by themselves establish boundary
coverage.

Source: `E-STAGEIII-PASS`.

## Final cold-start convergence audit

The final cumulative Stage III dataset contains 4345 unique exact points. The
audit uses 3213 trusted gapped FFLO points, classified as 2290 cFFLO and 923
tFFLO. On a fixed normalized (401\times401) audit grid, the last three map
changes are 0.000471, 0.000908, and 0.000488; topology-boundary shift (p_{95})
values are 0.000815, 0.002625, and 0.000871; final coverage (p_{95}) is
0.006185; and trusted topology surprise values are 0.0119, 0.0171, and 0.0126.
The final direct/bracket support fraction is 1.0, and both significant boundary
and topological-region component counts remain one over the last three states.

Source: `E-STAGEIII-CONVERGENCE`.

## Sensitivity and provenance caveats

The configured (k=8) neighbor audit passes the nominal 0.00625 coverage
threshold. The (k=6) sensitivity gives 0.006269, slightly above the threshold,
whereas (k=12) gives 0.005913. Map and component structure remain stable.
The 2026-08-13 scientific-evidence archive recovery provides the final NPZ/CSV,
per-iteration convergence tables, figure data, plotting source, report, and
original figures. Redundant intermediate cumulative NPZ checkpoints were
intentionally omitted, so the publication panel can be regenerated from the
retained evidence but the entire audit cannot be replayed from every raw
checkpoint locally.

The frozen decision JSON names the NPZ in `source_dataset` while its recorded
SHA-256 is the correct hash of `dataset_iter018.csv`. The per-iteration
provenance table names the CSV correctly, and all 51 NPZ arrays agree with the
paired CSV. This metadata-label discrepancy is recorded without editing the
frozen decision.

Source: `E-STAGEIII-CONVERGENCE`; repository provenance audit.
