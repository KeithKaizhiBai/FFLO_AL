# Figure 1 v5 branch and workflow semantics

## Report-ready interpretation

The central $F(\Delta,q)$ surface is a schematic physical state manifold. Its
normal, uniform-SC, cFFLO, and tFFLO markers represent characteristic minima
encountered while active learning explores parameter space. In particular,
the cFFLO and tFFLO markers should not be interpreted as simultaneously stable
minima at a single fixed parameter point.

Free-energy depth establishes thermodynamic preference, not topology. The
cFFLO/tFFLO distinction is made separately: a trusted gapped bulk is first
established, after which the Pfaffian sign test assigns $\nu=0$ or $\nu=1$.
The red branches in the tFFLO mini-panel are in-gap boundary/surface states,
not gapless bulk bands.

The surrounding machine-learning loop schedules physical calculations. The
exact BdG oracle supplies thermodynamic and spectral/topological evidence; the
surrogate predicts responses and uncertainty; acquisition ranks candidates;
and selected candidates return as new exact calls. No surrogate output is
treated as a replacement for the physics oracle.

## Figure-gate state

The v5 geometry and scientific-semantics lint passes with zero hard failures,
but `GATE_FIG1_V5_APPROVAL` remains human-controlled. No manuscript LaTeX or
Figure 2 work is authorized by this note.
