# Stage V iter002 reward-model hotfix

This folder contains a small patch script for the Stage V run:

```text
stagev_acqv2_boundary_support_learned_residual_3d_v1
```

## Problem

The first HPC run completed:

```text
iter000 -> dataset_iter001
iter001 -> dataset_iter002
```

Then acquisition for iter002 failed because the learned reward model was trained
with `selection_probability`, while pre-selection candidate scoring does not
have that field yet.

## Upload

Upload this folder into the Stage V package directory on the cluster, for
example:

```text
~/bkz/Fu_FFLO/stagev_acqv2_boundary_support_learned_residual_3d_v1_hpc/
```

## Apply

From the Stage V package root:

```bash
python hpc_hotfix_scripts/stagev_acqv2_iter002_reward_model_patch/apply_stagev_acqv2_iter002_hotfix.py
```

or:

```bash
bash hpc_hotfix_scripts/stagev_acqv2_iter002_reward_model_patch/apply_stagev_acqv2_iter002_hotfix.sh
```

The script auto-detects the package root, patches
`ml_phase/stagev_acqv2.py`, and runs:

```bash
python -m py_compile ml_phase/stagev_acqv2.py scripts/stagev_acqv2_select.py
```

## Resume

Keep the completed `dataset_iter002` and resume from iter002:

```bash
export CONFIRM_STAGEV_PRODUCTION=1
START_ITER=2 nohup bash scripts/resume_stagev_acqv2_full_loop.sh \
  > stagev_acqv2_boundary_support_learned_residual_3d_v1_resume_iter002.nohup.log 2>&1 &
```
