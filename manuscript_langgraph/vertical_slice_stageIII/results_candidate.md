# Results candidate

## Topology-aware cold-start reconstruction

We next incorporated topology into the exact-label hierarchy without altering
the thermodynamic normal, uniform-SC, and FFLO definitions. For each
superconducting point, the exact oracle evaluates the Pfaffians $P_0$ and
$P_\pi$ together with the minimum quasiparticle gap over the full Brillouin
zone. The $\mathbb{Z}_2$ label is assigned only to trusted gapped states, so a nodal
spectrum and a numerically unresolved spectrum remain distinct outcomes. The
analytic Pfaffian convention agrees with the numerical Majorana-basis
calculation in all 104 non-boundary validation cases, while the CPU and GPU
bulk-gap calculations agree to $7.77\times10^{-16}$ at $N_k=2048$.

The final cold-start run contains 4345 unique exact samples, including 3213
trusted gapped FFLO points. Of these, 2290 are classified as cFFLO and 923 as
tFFLO. A no-future-leakage convergence audit on a fixed (401\times401) grid
finds final-three topology-map changes of 0.000471, 0.000908, and 0.000488 and
boundary-shift $p_{95}$ values of 0.000815, 0.002625, and 0.000871. The final
coverage $p_{95}=0.006185$ lies below the configured 0.00625 threshold, and
trusted topology surprise remains below 0.02 in all three transitions. The
significant boundary and topological-region component counts both remain one,
so the configured audit returns Decision A and does not request an additional
exact calculation.

This conclusion is tied to the configured audit rather than to an assertion
that every possible refinement has been exhausted. In particular, changing
the local-neighbor parameter from $k=8$ to $k=6$ gives coverage
$p_{95}=0.006269$, slightly above the nominal threshold, whereas $k=12$
gives 0.005913; the phase-map and component structure remain stable. Moreover,
local opposite-$\mathbb{Z}_2$ brackets diagnose boundary support but are not themselves
identified with the final topology contour.
