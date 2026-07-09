# HPC Fetch And Audit Template

Use this template when Codex should inspect remote H100 results and fetch a
small, targeted subset for local analysis.

## Remote Context

```text
ssh alias:
remote project root:
package directory:
run_id:
output_root:
```

## Allowed Remote Commands

Use read-only commands by default:

```text
ssh
ls
find
du -sh
tail
cat small files
squeue
sacct
```

## Fetch Policy

Estimate size with `du -sh` before transfer.

Prefer fetching:

```text
reports/
logs/
manifests/
small CSV files
final figures
final iteration folder if small enough
```

Do not fetch large `.npy`, `.npz`, `.h5`, `.pt`, `.pkl`, or raw data unless
explicitly requested.

## Audit Questions

1. Did the remote run finish?
2. What is the latest dataset iteration?
3. Are final shards, merged files, trusted files, and logs present?
4. Are there Slurm failures or transient accounting races?
5. What local report should be generated?
