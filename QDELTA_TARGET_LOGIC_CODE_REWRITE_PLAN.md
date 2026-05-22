# Q/Delta Target-Logic Code Rewrite Plan

This plan rewrites the current risk-labeling exact oracle into the target
physics-aware confirmation logic.

Current behavior:

```text
q_edge_hit is detected but does not trigger q-window expansion.
delta_boundary_ambiguous is detected but does not trigger Delta refinement.
normal-state points with Delta_opt = 0 can still be marked q_edge_hit.
```

Target behavior:

```text
1. coarse exact scan;
2. classify whether q is physically meaningful;
3. if superconducting and q hits boundary, expand q and rerun;
4. if near normal/SC boundary, refine Delta and rerun;
5. only trusted confirmed points enter the trusted exact dataset;
6. unresolved points go to rerun_points.csv with clear reason.
```

## 1. Core Principle

Do not treat \(q_{\mathrm{opt}}\) as physically meaningful when
\(\Delta_{\mathrm{opt}}\) is zero or below threshold.

Use this decision order:

```text
1. Determine whether Delta_opt indicates normal or superconducting state.
2. If normal:
       q_status = not_applicable
       q_edge_hit = false
       only Delta-boundary confirmation matters.
3. If superconducting:
       q_status = active
       q_edge_hit can trigger q-window expansion.
4. If Delta is close to the normal/SC threshold:
       perform Delta refinement regardless of q status.
```

## 2. Status Fields

Add explicit status fields instead of relying only on bit-coded
`exact_status_code`.

Required output arrays from `ml_phase/exact_oracle.py`:

```text
kT
JA
eta
q_opt
delta_opt
ic_plus
ic_minus

phase_candidate
q_status
q_min
q_max
n_q
q_index
q_edge_distance
q_edge_hit
q_refinement_level
q_expanded
q_unresolved

delta_status
delta_min
delta_max
n_delta
n_delta_refined
delta_refinement_level
delta_boundary_ambiguous
delta_refined
delta_unresolved
free_energy_gap_to_normal

exact_status_code
exact_status_name
trusted_exact
```

Recommended integer conventions:

```text
phase_candidate:
    0 normal
    1 superconducting
    2 ambiguous

q_status:
    0 not_applicable
    1 active
    2 edge_hit
    3 expanded_confirmed
    4 unresolved

delta_status:
    0 stable
    1 boundary_ambiguous
    2 refined_confirmed
    3 unresolved
```

Recommended bit-coded `exact_status_code`:

```text
0   trusted
1   q_edge_unresolved
2   delta_boundary_unresolved
4   nonfinite_output
8   max_q_refinement_reached
16  max_delta_refinement_reached
32  normal_state_q_not_applicable
```

`trusted_exact` should be true only if the point is numerically acceptable for
training.

## 3. Refactor Exact Oracle Internals

Target file:

```text
ml_phase/exact_oracle.py
```

Introduce a point-level internal result dataclass:

```python
@dataclass
class PointOracleResult:
    kT: float
    JA: float
    eta: float
    q_opt: float
    delta_opt: float
    ic_plus: float
    ic_minus: float
    omega_global: float
    q_min: float
    q_max: float
    n_q: int
    q_index: int
    delta_min: float
    delta_max: float
    n_delta: int
```

Introduce helper functions:

```python
def evaluate_one_point_once(kT, JA, cfg, device) -> PointOracleResult:
    """Run one exact solve with the cfg's current q/Delta grid."""

def classify_point_status(result, cfg, ml_cfg_or_thresholds) -> dict:
    """Return phase_candidate, q_status, delta_status, flags."""

def expand_q_config(cfg, q_opt, hit_side, expand_factor, pad_steps, q_max_abs):
    """Return a copied EtaPhaseConfig with expanded q range."""

def refine_delta_config(cfg, delta_opt, half_width, n_delta_refined):
    """Return a copied EtaPhaseConfig with narrower/finer Delta interval."""
```

Do not mutate a shared config in-place inside an iteration over points. Build
per-point copied configs for expansion/refinement.

## 4. Target q Logic

Inside `evaluate_points`, for each point:

```text
1. Run evaluate_one_point_once using default q/Delta grid.
2. If result is nonfinite:
       mark nonfinite and stop.
3. If delta_opt < DELTA_EPS and not Delta-boundary ambiguous:
       phase_candidate = normal
       q_status = not_applicable
       q_edge_hit = false
       trusted_exact = true
       stop.
4. If delta_opt < DELTA_EPS but Delta-boundary ambiguous:
       q_status = not_applicable
       skip q expansion
       proceed to Delta refinement.
5. If delta_opt >= DELTA_EPS:
       q_status = active
       check q_edge_hit.
6. If q_edge_hit:
       expand q window and rerun.
7. Repeat q expansion until:
       q_edge_hit is false,
       or max_q_refinements reached,
       or q range reaches q_max_abs.
8. If still q_edge_hit:
       q_status = unresolved
       trusted_exact = false.
```

Important: if the first run gives `Delta_opt = 0` and `q_opt = q_min`, this
should usually be treated as normal-state `q_status = not_applicable`, not
physical q truncation.

## 5. Target Delta Logic

Delta refinement should trigger when:

```text
abs(delta_opt - DELTA_EPS) <= delta_boundary_margin
or abs(free_energy_gap_to_normal) <= free_energy_ambiguity_tol
or phase changes across nearby coarse samples if that diagnostic is available
```

Inside each point:

```text
1. After initial solve, check Delta ambiguity.
2. If not ambiguous:
       delta_status = stable.
3. If ambiguous:
       build refined Delta interval:
           center = max(delta_opt, DELTA_EPS)
           lo = max(0, center - delta_refine_half_width)
           hi = min(delta_max_global, center + delta_refine_half_width)
           n_delta = n_delta_refined
4. Rerun exact solve with same q policy and refined Delta grid.
5. If refined result remains ambiguous but max_delta_refinements not reached:
       refine again with narrower interval or larger n_delta.
6. If still ambiguous:
       delta_status = unresolved
       trusted_exact = false unless explicitly allowed.
7. If refined result is stable:
       delta_status = refined_confirmed
       use refined delta_opt and observables.
```

For consistency, q expansion and Delta refinement should be ordered as:

```text
coarse solve
-> decide normal vs SC relevance
-> q expansion if SC and q hits boundary
-> Delta refinement if Delta boundary is ambiguous
-> if Delta refinement changes normal/SC classification, re-evaluate whether q
   is applicable
```

## 6. CLI Changes

Add to `ml_phase/exact_oracle.py`:

```text
--enable-q-expansion
--q-expand-factor 1.5
--q-expand-pad-steps 50
--q-max-abs 3.141592653589793
--max-q-refinements 3

--enable-delta-refinement
--delta-refine-half-width 0.03
--n-delta-refined 300
--max-delta-refinements 2

--allow-ambiguous-output
```

Default should be conservative:

```text
q/delta metadata always recorded;
automatic q/delta reruns enabled for production H100 scripts after smoke tests;
ambiguous points not trusted by default.
```

## 7. SLURM Script Changes

Target file:

```text
scripts/slurm_exact_oracle_array.sh
```

Add environment-controlled flags:

```bash
Q_EXPANSION_FLAG="${Q_EXPANSION_FLAG:---enable-q-expansion}"
DELTA_REFINEMENT_FLAG="${DELTA_REFINEMENT_FLAG:---enable-delta-refinement}"
MAX_Q_REFINEMENTS="${MAX_Q_REFINEMENTS:-3}"
MAX_DELTA_REFINEMENTS="${MAX_DELTA_REFINEMENTS:-2}"
N_DELTA_REFINED="${N_DELTA_REFINED:-300}"
```

Pass them into:

```bash
"${PYTHON_BIN}" -m ml_phase.exact_oracle ...
```

For debugging, allow:

```bash
export Q_EXPANSION_FLAG=
export DELTA_REFINEMENT_FLAG=
```

## 8. HPC Merge and Trusted Filtering

Target file:

```text
ml_phase/hpc.py
```

Add:

```python
def split_trusted_and_rerun(merged: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    trusted = trusted_exact == 1 and exact_status_code == 0
    rerun = not trusted
```

Write:

```text
exact_merged_iterXXX.npz
exact_trusted_iterXXX.npz
rerun_points.csv
```

`rerun_points.csv` should include:

```text
kT
JA
q_opt
delta_opt
phase_candidate
q_status
delta_status
exact_status_code
reason
recommended_action
```

Reasons:

```text
q_edge_unresolved
delta_boundary_unresolved
nonfinite_output
normal_q_not_applicable
max_q_refinement_reached
max_delta_refinement_reached
```

`normal_q_not_applicable` is informational and should not by itself force a
rerun if Delta is stable.

## 9. Active Refine Append Logic

Target file:

```text
ml_phase/active_refine.py
```

Current future loop must append only trusted points:

```text
if exact_trusted_iterXXX.npz exists:
    append that
else:
    refuse to append exact_merged_iterXXX.npz unless --allow-ambiguous-append
```

Do not train on unresolved q-edge or Delta-ambiguous labels by default.

## 10. Report Changes

Target files:

```text
report/active_learning_phase_boundary_report.tex
ml_phase/report_builder.py
```

Add report fields:

```text
trusted exact points
rerun-required points
q expanded confirmed count
q unresolved count
Delta refined confirmed count
Delta unresolved count
normal q-not-applicable count
```

Interpretation text:

```text
High q-unresolved rate means the q window remains too narrow.
High Delta-unresolved rate means normal/SC boundary needs finer Delta scans.
High normal q-not-applicable rate is expected in normal regions and should not
be interpreted as FFLO q truncation.
```

## 11. Tests

Add lightweight tests or scripts under:

```text
tests/ or scripts/dev_check_qdelta_logic.py
```

Minimum checks:

```text
1. Synthetic normal point:
       delta_opt < DELTA_EPS, q_opt = q_min
       expected q_status = not_applicable
       q_edge_hit = false
       trusted if Delta stable

2. Synthetic SC q-edge point:
       delta_opt > DELTA_EPS, q_opt = q_min
       expected q expansion attempted

3. Synthetic Delta-boundary point:
       delta_opt ~= DELTA_EPS
       expected Delta refinement attempted

4. Merge synthetic statuses:
       trusted_exact filters correctly
       rerun_points.csv reasons are correct
```

Full exact tests on H100:

```text
1. one normal point;
2. one trusted SC point near q interior;
3. one high-JA point expected to trigger q expansion;
4. one normal/SC boundary point expected to trigger Delta refinement.
```

## 12. Acceptance Criteria

The rewrite is accepted only when:

```text
1. 32-point H100 smoke run completes.
2. normal points are not mislabeled as physical q-edge failures.
3. q-edge SC points attempt q expansion.
4. Delta-boundary points attempt Delta refinement.
5. exact_trusted_iter000.npz and rerun_points.csv are both generated.
6. report compiles locally after download.
7. trusted exact ratio is reported.
```

After this is done, implement the actual multi-iteration active-learning loop.

