# Stage IV 3D Topology-Aware Cold-Start Full Loop

Run id:

```text
active_phase_topology_3d_t_ja_mu_from_scratch_v1
```

This package is self-contained for the Stage IV cold-start 3D active-learning run over
`(kBT/t, J_A/t, mu/t)`. It does not import Stage III datasets or checkpoints.

For the exact command checklist, see:

```text
HPC_COMMANDS_STAGEIV_3D.md
```

## Submit

```bash
cd ~/bkz/Fu_FFLO/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624
unset OUTPUT_ROOT RUN_ID CONFIG_JSON
export CONFIRM_STAGEIV_FULL_LOOP=1
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
nohup bash scripts/submit_stageiv_3d_full_loop.sh > active_phase_topology_3d_t_ja_mu_from_scratch_v1.nohup.log 2>&1 &
```

The submit script uses `--exclude=gpuh01` and the Slurm script also refuses to run if
the runtime hostname resolves to `gpuh01`.

## Collect

```bash
bash scripts/collect_stageiv_3d_results.sh
```

Run collection after the Slurm loop finishes.  The post-run reports should then
be built from the returned/extracted output directory.

## Operations

```bash
bash scripts/check_stageiv_3d_submit_ready.sh
bash scripts/run_stageiv_3d_preflight.sh
bash scripts/monitor_stageiv_3d_run.sh
bash scripts/check_stageiv_3d_hpc_status.sh
RETURN_PATH=ML_Phase_StageIV_Topology3D bash scripts/check_stageiv_3d_return_bundle.sh
START_ITER=7 bash scripts/resume_stageiv_3d_full_loop.sh
ITER=0 FAILED_RANKS=3 bash scripts/recover_stageiv_failed_exact_iter.sh
bash scripts/inspect_stageiv_3d_checkpoint.sh
bash scripts/inspect_stageiv_failed_task.sh <job_id>
bash scripts/collect_stageiv_3d_results.sh
bash scripts/build_stageiv_3d_postrun_summary.sh
bash scripts/build_stageiv_3d_convergence_audit.sh
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_hidden_slice_audit.sh
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_all_postrun_reports.sh
```

## Notes

- Production seed size: 1024 scrambled Sobol points.
- Acquisition batch size: 256.
- Acquisition batches: 24 after seed.
- Exact oracle: robust incremental q-window with basin-level rank-and-cap K3.
- Topology diagnostics: Pfaffian Z2 plus full-BZ bulk gap.
