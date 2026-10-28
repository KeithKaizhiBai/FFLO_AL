# Stage V-v4 Minimum Audit Download

Date: 2026-07-11

## Question

What is the minimum Stage V-v4 remote result set needed to build a formal
return report and convergence/acquisition audit without downloading raw H100
shards and full candidate pools?

## Short Answer

Download every cumulative dataset CSV, the final NPZ, per-iteration selected
metadata, graph node/edge summaries, selection/reward/merge summaries, reward
history, final learning state, configurations, provenance, and controller
logs. This is sufficient to reconstruct cumulative thermodynamic and topology
states, boundary coverage and shift proxies, trusted surprise, proposal/action
efficiency, graph evolution, and online value-model behavior.

## Downloaded Artifact

```text
downloads/stagev_v4_min_report_audit_20260711/
stagev_v4_min_report_audit_data.tar.gz
sha256 = ac69a6d20bb3520f43bca4e00bd4883e18b26e6587960f62a2b7795824756ee2
compressed size = approximately 53 MiB
tar members = 886
```

The stream was created read-only over SSH. No remote archive was created.

## Scope Boundary

Raw exact shards, branch-candidate files, full candidate pools, graph
descriptor tables, and per-task Slurm logs were excluded. Those artifacts are
needed only for low-level shard forensics, detailed branch replay, or a full
candidate-level counterfactual acquisition reconstruction. Their exclusion
does not change any physical definition or numerical convention.

At download time the production loop had completed through exact iter096 and
written `dataset_iter097`; no formal Stage V-v4 return report or convergence
audit existed remotely.
