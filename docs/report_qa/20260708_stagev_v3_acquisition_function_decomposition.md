# Stage V-v3 Acquisition Function Decomposition

## Question

Is Stage V-v3 learning the acquisition function from scratch?

## Short Answer

Stage V-v3 is a strict cold-start run with respect to production data and
learning state. It does not initialize its training set, selected coordinates,
reward history, value models, lambda state, or alpha state from Stage IV,
Stage V-v1, Stage V-v2, V2-tail, `dataset_iter097`, or `dataset_iter101`.

However, it does not learn the entire acquisition function from a blank
symbolic form. The acquisition function has an explicit physics/support base
score, and V3 learns an online residual/value correction from its own exact
feedback.

## Decomposition

For boundary head

```text
s in {NS, UF, P0, Ppi, gap}
```

the explicit base acquisition is:

```latex
A_s^{(0)}(x) = B_s(x) U_s(x) H_s(x)
```

where:

- `B_s`: boundary likelihood or residual proximity;
- `U_s`: model or surrogate uncertainty;
- `H_s`: support sparsity or fill-distance pressure.

For topology heads, Stage V-v3 adds sparse topology-surface support:

```latex
A_{P0/P\pi}^{v3}(x) \propto A_{P0/P\pi}^{(0)}(x) C_{\rm topo}(x)
```

where `C_topo` estimates how much a candidate would reduce local support
distance on sparse cFFLO/tFFLO topology-surface samples.

The learned residual enters as:

```latex
\ell_s(x) =
    \log A_s^{v3}(x) + \lambda_s g_s(\phi_s(x))
```

and the heads are combined by rank-normalized log-score aggregation with
head priority `alpha_s` and source-density correction.

## Training

The learned component `g_s` is fit only from Stage V-v3 selected-point metadata
and exact outcomes accumulated so far. The reward includes per-head terms for:

- local topology-coverage gain;
- p95 coverage-contributor gain;
- opposite-Z2 bracket creation or shortening;
- sparse-component support;
- residual proximity;
- trusted topology surprise;
- numerical reliability penalties.

`lambda_s` gates the learned residual. It is reduced or held when reward
prediction improves but topology coverage does not. `alpha_s` updates head
priority according to support, surprise, shift, and convergence deficits.

## Report Integration

The Stage V-v3 return report now includes:

- the explicit acquisition formulas;
- `tables/stagev3_acquisition_function_components.csv`;
- `tables/stagev3_acquisition_training_loop.csv`;
- a LaTeX section explaining that V3 learns the residual/value correction from
  scratch, not the entire acquisition law without physics assumptions.

## Caveat

This decomposition explains the acquisition design and training flow. It does
not change the Stage V-v3 decision: the run remains Decision C because topology
surface coverage and trusted topology surprise do not yet pass formal gates.
