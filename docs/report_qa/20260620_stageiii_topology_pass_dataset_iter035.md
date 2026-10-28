# Stage III Topology Pass on Frozen `dataset_iter035`

## Context

Stage III begins from the frozen thermodynamic active-learning dataset
`dataset_iter035`.  The thermodynamic labels are not modified:

```text
total = 7434
normal = 1867
uniform-SC = 715
FFLO = 4852
```

The run id is:

```text
topology_pass_dataset_iter035_v1
```

## What Was Computed

For existing superconducting exact points, the offline pass computed:

```text
P0
Ppi
Pfaffian product
dimensionless Pfaffian margin
full-Brillouin-zone bulk gap
k at minimum bulk gap
thermal gap ratio Eg / kBT
preliminary topology label
topology trust flag
```

The bulk gap was computed with the existing project BdG Hamiltonian builder,
using float64/complex128 batched eigensolves over the full Brillouin zone.

## Convention Check

The analytic Pfaffian was cross-checked against the numerical Pfaffian built
from the project BdG Hamiltonian in the current Nambu/Majorana convention.

Verified convention:

```text
P0  = (mu - t cos(q/2))^2 + Delta^2
      - alpha_y^2 sin^2(q/2)
      - (J_A cos(q/2) + alpha_z sin(q/2))^2

Ppi = (mu + t cos(q/2))^2 + Delta^2
      - alpha_y^2 sin^2(q/2)
      - (J_A cos(q/2) + alpha_z sin(q/2))^2
```

Validation result:

```text
analytic/numeric Pfaffian product-sign agreement = 104/104 non-boundary cases
CPU/GPU bulk-gap agreement at Nk=2048: max absolute difference = 7.77e-16
```

## Main Result

Output directory:

```text
reports/topology_pass_dataset_iter035_v1/
```

Topology counts:

```text
uniform-SC:
    trivial = 715
    topological = 0
    gapless = 0
    unresolved = 0

FFLO:
    trivial = 3127
    topological = 1515
    gapless = 195
    unresolved = 15
```

Trusted topology points:

```text
5357 / 5567 superconducting points
```

The 15 unresolved points are numerical-reliability states from the bulk-gap
trust rule.  They should not be interpreted as a physical gapless phase.  The
195 `gapless_SC` points are points where the local k-refined full-BZ bulk-gap
oracle resolved the spectrum as gapless under the current tolerance.

## Coverage Diagnostics

Trusted FFLO gapped points were used for Delaunay diagnostics:

```text
trusted FFLO gapped points = 4642
Z2-change candidate edges = 182
P0 sign-change candidate edges = 182
Ppi sign-change candidate edges = 0
large circumradius coverage-hole triangles = 447
```

These Delaunay edges are topology-boundary seeds and coverage diagnostics
only.  They are not final topological phase boundaries.

## Resource Decision

The 256-point pilot showed that local execution was appropriate:

```text
selected backend = gpu
Nk = 2048
bulk-gap scan runtime recorded in summary = about 5.3 seconds
complete full-pass script with local k refinement = about 70 seconds
```

The 4090 GPU was selected by measured pilot runtime after CPU/GPU agreement,
not by assumption.

## Caveats

```text
1. Do not modify or reinterpret the original normal / uniform-SC / FFLO labels.
2. Do not call sparse Delaunay edges final topology contours.
3. Do not treat topology_unresolved as gapless.
4. Do not use Pfaffian sign alone without the full bulk-gap status.
5. Parquet output was not written because pyarrow/fastparquet is not installed;
   CSV and NPZ outputs are available.
```

## Recommended Next Step

The result supports Case B:

```text
There are trusted topological and trivial FFLO points, but topology-boundary
coverage is still inherited from thermodynamic active learning and remains
insufficient for a final contour claim.
```

Next calculation should be a topology-aware acquisition/refinement pilot
targeting:

```text
small Pfaffian margin
small bulk gap
candidate Z2-change Delaunay edges
coverage-hole regions
```
