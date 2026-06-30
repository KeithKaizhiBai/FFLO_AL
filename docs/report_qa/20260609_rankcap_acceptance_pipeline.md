# Rank-and-Cap K3 Acceptance Pipeline

Date: 2026-06-09

Question: What is the current status of the one-pipeline rank-and-cap K3 acceptance test?

Answer:

- A single acceptance entrypoint was added: `scripts/run_local_refinement_rankcap_acceptance.py`.
- The pipeline compares only `baseline` and `rank_and_cap_k3`.
- It does not run k2, energy-window, branch reuse, Powell, adaptive boxes, GPU batching, Hamiltonian cache, mini AL, or full AL.
- Local report output is under `reports/local_refinement_rankcap_acceptance/`.
- Gate A uses the returned 32-point target-construction evidence and passes:
  - 32/32 fixed points covered.
  - `rank_and_cap_k3` selected target count max is 3.
  - mandatory overflow is still recorded, but rank-and-cap handles it without selected targets exceeding 3.
- Gate B and Gate C are pending because the real local-box regression has not been run locally.
- Current `acceptance_status` is `pending_hpc_regression`, not pass.
- The workflow is packaged at `hpc_packages/local_refinement_rankcap_acceptance/` and `hpc_packages/local_refinement_rankcap_acceptance.tar.gz`.
- The package task matrix contains exactly 64 tasks: 32 baseline and 32 `rank_and_cap_k3`.
- Shell scripts are LF-normalized and GPU Slurm scripts exclude `gpuh01`.

Current decision:

Do not enter one-iteration AL validation. Run the packaged HPC acceptance workflow first, then inspect the returned report.
