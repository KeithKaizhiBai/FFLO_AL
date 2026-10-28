# Stage V-v2 Topology Coverage Tail Submission Notes

## Scope

The shared prompt `shared_prompts/inbox/Stage_v2_Converage_tail_prompt.md`
was copied to `shared_prompts/active/CURRENT_TASK_PROMPT.md` and used to build
and submit a targeted topology-surface coverage tail.

This is not a from-scratch active-learning loop.  It does not change the
original Stage V-v2 production run, historical thermodynamic labels, topology
definitions, Hamiltonian, exact oracle, or Pfaffian convention.

## Source State

```text
source_run_id = stagev_v2_multihead_boundary_learning_3d_v1
source_dataset = dataset_iter097
source_exact_iteration = 96
formal_audit_decision = Decision C
limiting_factor = topology_surface_coverage
topology_surface_coverage_p95_final = 0.036647
coverage_reference_threshold = 0.00625
```

The phase map, topology-map proxy, and surface-shift proxy were already stable
in the formal audit.  The remaining issue was sparse exact support near parts
of the cFFLO/tFFLO topology surface.

## Tail Design

The tail package uses the formal-audit opposite-Z2 topology bracket table to
select 256 targeted exact points:

```text
run_id = stagev2_topology_surface_coverage_tail_v1
output_root = ML_Phase_StageV_V2_TopologyCoverageTail
tail_exact_iterations = 097..100
points_per_batch = 64
total_new_exact_points = 256
source table = stagev_v2_final_topology_brackets.csv
```

The selected points include both P0-crossing and Ppi-crossing topology
brackets.  The selected set is written as explicit `kT,JA,mu` CSV shards, with
non-empty `proposal_source` metadata for traceability.

## HPC Submission

Remote package:

```text
/public_hw/home/sci_bfu/bkz/Fu_FFLO/stagev2_topology_surface_coverage_tail_v1_hpc
```

First submitted exact array:

```text
job_id = 89488
initial_status = RUNNING
initial_nodes = gpuh02,gpuh05
excluded_nodes = gpuh01,gpuh14
```

Initial shard logs showed CUDA available and the exact oracle entering
`robust_incremental` mode with topology classification enabled.

## Monitor Commands

```bash
ssh GBU-H100 "cd /public_hw/home/sci_bfu/bkz/Fu_FFLO/stagev2_topology_surface_coverage_tail_v1_hpc && tail -f stagev2_topology_surface_coverage_tail_v1.nohup.log"
ssh GBU-H100 "/opt/gridview/slurm-2.3.3/bin/squeue -j 89488"
ssh GBU-H100 "cd /public_hw/home/sci_bfu/bkz/Fu_FFLO/stagev2_topology_surface_coverage_tail_v1_hpc && bash scripts/status_stagev2_topology_coverage_tail.sh"
```

## Next Audit

After the tail reaches `dataset_iter101`, collect the result bundle and rerun
the topology-surface coverage audit.  The key question is whether the
tail-augmented support reduces coverage p95 toward the 0.00625 reference
threshold without changing the already stable main topology map.
