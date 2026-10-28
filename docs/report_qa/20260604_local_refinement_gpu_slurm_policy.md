# 2026-06-04 Local Refinement GPU Slurm Policy

## Question

Why did the local-refinement refactor remain blocked after local packaging and
preflight passed, and does it require another upload/compute run?

## Answer

The remaining blocker is not a local import-path or writable-directory problem.
Those were fixed by making runtime outputs live under package-local `RUN_ROOT`
directories and by verifying nested package entry points before upload.

The unresolved gate is the Stage 2-4 variant-suite GPU fixed-point regression.
The full runbook is incomplete until the returned variant-suite archive is
imported locally with:

```text
gate_status = pass
import_status = pass
performance_report_status = pass
```

Because the cluster has a known problematic node, `gpuh01`, whose NVIDIA
driver is too old for the active PyTorch CUDA runtime, the upload-set verifier
now checks every nested GPU Slurm script for:

```text
#SBATCH --exclude=gpuh01
torch.empty(1, device="cuda")
cuda_runtime_probe=pass
```

This confirms that the current handoff package excludes the known bad node and
performs a real CUDA runtime probe before starting expensive fixed-point work.
The current upload set checks 7 GPU Slurm scripts and reports 0 policy
violations.

This is a packaging and HPC safety policy only.  It does not change the
Hamiltonian, phase labels, q/Delta refinement safeguards, exact-oracle
thresholds, branch reuse, adaptive boxes, GPU batching, or Hamiltonian cache
behavior.

The next required action is to upload:

```text
hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz
```

Then on the cluster run:

```bash
python verify_upload_set.py
tar -xzf archives/local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_local_refinement_fixed_point_regression.sh
```

Return:

```text
$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run/local_refinement_refactor_variant_suite_results.tar.gz
```

and import it locally with:

```bash
python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```
