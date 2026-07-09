# Stage IV 3D Repackage Identity Guard

Date: 2026-06-24

## Question

The submitted Stage IV Slurm job used a Stage IV job name, but the logs showed:

```text
run_id=active_phase_topology_from_scratch_full_loop_v2
iteration=11
ML_Phase_512_TopoTrivial_FullLoop/active_runs/...
```

Was this the intended 3D phase-diagram run?

## Answer

No.  The job executed inside the Stage IV upload directory, but stale shell
identity variables or an older script path redirected the output namespace to a
previous topology full-loop run.  The computation should be preserved as an
older-namespace result, but it is not valid evidence for the Stage IV 3D
production objective.

## Fix

The Stage IV 3D package was rebuilt with package-level frozen identity guards:

```text
expected output_root = ML_Phase_StageIV_Topology3D
expected run_id      = active_phase_topology_3d_t_ja_mu_from_scratch_v1
expected config      = configs/stageiv_3d_production.json
```

Generated shell scripts now reject conflicting `OUTPUT_ROOT`, `RUN_ID`, or
`CONFIG_JSON` values instead of silently accepting them.

The final upload archive was renamed to avoid directory/archive conflicts with
earlier Stage IV attempts:

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624.tar.gz
```

## Validation

```text
package_validation_status = pass
stale_2d_marker_scan = []
package_sha256 = 4799ecfca7ab16c8a340d731dfb74cf0ea44502b2c4582b59523e65df00fcbf4
```

The package keeps `gpuh01` excluded through both Slurm `--exclude=gpuh01` and a
runtime hostname guard.

## Operational Note

Before submitting the rebuilt package, clear stale shell identity variables:

```bash
cd ~/bkz/Fu_FFLO/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624
unset OUTPUT_ROOT RUN_ID CONFIG_JSON
export CONFIRM_STAGEIV_FULL_LOOP=1
nohup bash scripts/submit_stageiv_3d_full_loop.sh > active_phase_topology_3d_t_ja_mu_from_scratch_v1.nohup.log 2>&1 &
```

This fix does not change thermodynamic phase criteria, topology formulas,
acquisition rules, StopController thresholds, exact-oracle tolerances, or
rank-and-cap local refinement behavior.
