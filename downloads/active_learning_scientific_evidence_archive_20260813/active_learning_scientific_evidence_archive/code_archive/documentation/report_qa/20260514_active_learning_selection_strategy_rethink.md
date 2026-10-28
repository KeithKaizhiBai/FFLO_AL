# Active-Learning Selection Strategy Rethink

## Question

Why should the next active-learning revision remove midpoint-based point
selection?

## Short Answer

Midpoint selection is a deterministic bisection rule on already known exact
brackets. It can refine a bracket geometrically, but it does not let the
trained ML model decide where a new exact calculation would most reduce
uncertainty, error, or boundary ambiguity. For the next stage, midpoint quotas
should be removed and replaced by ML-guided boundary proposals.

## Technical Notes

The current hybrid workflow contains two point-selection routes:

```text
1. dense-grid ML acquisition:
   trained MLP predictions -> acquisition score -> ranked candidates

2. boundary-local midpoint refinement:
   accepted exact brackets -> geometric midpoint -> exact oracle
```

The second route is based on exact labels, not ML predictions. This is useful
as a numerical bisection operation, but it conflicts with the active-learning
goal when it consumes a large fixed fraction of the exact-call budget. The
model should provide more of the sampling signal.

## Radius Reinterpretation

The radius should be treated as a sampling-density and redundancy-control
parameter, not as a physical convergence criterion. A fixed radius can prevent
repeat points, but it can also produce rigid selection patterns or suppress
useful candidates near rapidly changing regions.

Useful alternatives include:

```text
1. adaptive radius r(x), smaller near high uncertainty or high predicted
   boundary gradient and larger in flat regions;
2. soft diversity penalties instead of a hard exclusion radius;
3. batch-level diversity optimization rather than one-by-one greedy selection.
```

## Brainstormed Implementation Directions

### Option A. Pure ML acquisition with adaptive radius

Remove midpoint pools. Score all admissible dense-grid candidates using the
trained model, then use a local radius such as

\[
    r(x)=r_{\min}
    +
    \frac{r_{\max}-r_{\min}}{1+\alpha U(x)+\beta G(x)}.
\]

This keeps ML as the selection driver while still controlling repeated points.

### Option B. Boundary probability field

Build a continuous boundary-probability field from classifier probabilities
and regression threshold scores:

\[
    P_{\partial}(x)
    =
    P_{\mathrm{cls,boundary}}(x)
    +
    \lambda_\Delta B_\Delta(x)
    +
    \lambda_q B_q(x).
\]

Candidate points are selected from ridges of this learned boundary field,
including regions not yet bracketed by exact data.

### Option C. Learned utility model

Log candidate features and later exact outcomes, then train a lightweight
secondary model to estimate the expected gain of selecting a candidate:

\[
    \widehat{U}_{\mathrm{gain}}(x)
    \approx
    \Delta \mathrm{error}(x)
    \quad\text{or}\quad
    \Delta \mathrm{boundary}(x).
\]

This would make acquisition weights data-calibrated rather than purely hand
chosen.

### Option D. Batch-level information and diversity selection

Select the whole batch by optimizing a score such as

\[
    \mathcal{B}^{\star}
    =
    \arg\max_{\mathcal{B}}
    \sum_{x\in\mathcal{B}}S_{\mathrm{ML}}(x)
    -
    \gamma
    \sum_{x_i,x_j\in\mathcal{B}}K(x_i,x_j).
\]

This avoids relying on a single hard radius threshold while still discouraging
near-duplicate exact calls.

## Report Use

Use this material when explaining why the project is moving away from
geometric midpoint refinement toward a more ML-driven active-learning
selection policy.

## Implementation Update on 2026-05-14

The code path has been changed from the old hybrid selector to an
acquisition-only selector.

Current behavior:

```text
1. selected_points.csv is generated only from dense-grid ML acquisition;
2. selected_points_by_pool.csv records selection_source = acquisition;
3. boundary extraction is diagnostic-only;
4. targeted_refinement_points.csv is written with zero rows;
5. boundary_refinement_mode = diagnostic keeps boundary summaries but does not
   create exact-call targets;
6. boundary_refinement_mode = hybrid or local raises an error because those
   modes depended on midpoint-based selection.
```

The rigid distance radius has also been softened. Exact duplicate coordinates
are still forbidden, but nearby non-identical dense-grid points are multiplied
by an observation-repulsion factor instead of being removed by a hard
`existing_min_dist` cutoff. Within a batch, a separate batch-repulsion factor
discourages clustered selected points without imposing a single hard minimum
distance.
