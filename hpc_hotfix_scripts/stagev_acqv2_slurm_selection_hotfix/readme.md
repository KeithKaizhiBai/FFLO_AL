# Stage V Slurm-selection hotfix

This hotfix changes the Stage V control flow so that acquisition selection is
submitted as a Slurm job instead of being run on the login node.

## Why

The Stage V run reached:

```text
latest good dataset = dataset_iter014
next exact iteration = 14
```

The exact Slurm jobs completed normally.  The outer control script then printed
`Terminated` before creating `iter014`, which is consistent with the login node
killing the CPU-heavy `stagev_acqv2_select.py` selection step.

## What the patch changes

The patch creates:

```text
scripts/slurm_stagev_acqv2_select.sh
```

and modifies:

```text
scripts/submit_stagev_acqv2_full_loop.sh
```

The login node now only submits and waits for a selection job:

```text
sbatch scripts/slurm_stagev_acqv2_select.sh
```

Then it submits the exact array as before.

## Apply

Upload this folder into the Stage V package root, then run:

```bash
python hpc_hotfix_scripts/stagev_acqv2_slurm_selection_hotfix/apply_stagev_acqv2_slurm_selection_hotfix.py
```

If `unzip` creates `stagev_acqv2_slurm_selection_hotfix/` directly under the
package root, run:

```bash
python stagev_acqv2_slurm_selection_hotfix/apply_stagev_acqv2_slurm_selection_hotfix.py
```

Or, if the Python file itself was uploaded directly into the package root:

```bash
python ./apply_stagev_acqv2_slurm_selection_hotfix.py
```

The patch first tries an exact block replacement.  If the submit script differs
slightly in whitespace or line wrapping, it falls back to a marker-based
replacement around the `stagev_acqv2_select.py` selection block.

## Resume from current state

After the patch, continue from the latest good dataset:

```bash
export CONFIRM_STAGEV_PRODUCTION=1
START_ITER=14 FINAL_EXACT_ITER=17 nohup bash scripts/resume_stagev_acqv2_full_loop.sh \
  > stagev_resume_iter014_to017_slurmselect.nohup.log 2>&1 &
```

## Notes

- This hotfix does not change the exact oracle.
- This hotfix does not change thermodynamic or topology definitions.
- Slurm scripts exclude `gpuh01,gpuh14`.
- Selection uses `NV_H100` with one GPU allocation to guarantee a compute node.
