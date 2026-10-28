# Reliability-aware active learning for FFLO phase spaces

> PRL-integrated publication-form candidate; human approval pending.

## Abstract

Resolving finite-momentum superconducting phases across expanding parameter spaces requires both reliable free-energy minimization and controlled use of expensive exact calculations. We retain an audited Bogoliubov-de Gennes oracle as the sole source of thermodynamic and topological labels, while active learning only schedules the next exact evaluations. This separation yields an audited two-dimensional reconstruction of Normal, uniform-superconducting, and FFLO structure while keeping a hard-risk numerical layer explicit. On trusted gapped FFLO states, full-zone gap and Pfaffian diagnostics resolve connected cFFLO/tFFLO structure that passes the configured k=8 audit with a retained k=6 sensitivity. The same framework exposes thermodynamic and topology-aware structure in (kBT/t, JA/t, mu/t), although the strict three-dimensional topology convergence criteria are not all closed and no complete topology atlas is claimed.

## Introduction

The computational problem considered here is to map superconducting states when the equilibrium condensate may select both a finite pairing amplitude and a finite center-of-mass momentum. In the present altermagnetic BdG model, thermodynamic competition in (Delta,q) must be resolved together with numerical reliability and, inside the trusted gapped FFLO sector, a Z2 topology diagnosis. Exhaustive sampling becomes increasingly costly when each parameter point requires a free-energy landscape, branch refinement, and spectral checks, especially after the domain is extended beyond two coordinates. A data-driven scheduler can focus exact calls, but it would compromise the physical interpretation if surrogate predictions were promoted to final phase or topology labels. We therefore separate an exact BdG authority from reliability-aware active learning: the oracle determines accepted observations, whereas an ensemble acquisition rule selects only the next micro-batch. The resulting evidence chain comprises an audited two-dimensional thermodynamic reconstruction, a topology-resolved cFFLO/tFFLO result with an explicit sensitivity test, and a higher-dimensional extension whose strict topology closure remains incomplete. The main text first defines this division of labor, then follows the thermodynamic, topology-resolved, and three-dimensional results before closing with the limits of learned acquisition and a bounded world-model outlook.

## Integrated Results and Methods

### Exact oracle and physical labels

For external coordinates x, the oracle uses the stated FFLO Nambu basis and block Hamiltonian. At every exact call it constructs F(Delta,q), refines competing minima, and accepts the lowest reliable state only after comparison with the normal reference. The frozen thresholds separate Normal, uniform-SC, and FFLO observations; phase labels remain distinct from numerical states. Edge minima trigger adaptive q-window expansion and shallow Normal-SC competition triggers local Delta refinement, while unresolved outcomes are withheld from clean physical training labels. Ensemble surrogates rank the next exact micro-batch from accepted observations; they are schedulers rather than physics solvers. Figure 1 is a schematic hierarchy and is not itself evidence that any convergence gate has closed.

[Inline FIG-01-PIPELINE; frozen caption unchanged.]



### Audited thermodynamic reconstruction

The two-dimensional loop begins from 512 selected exact calls, with 506 labels retained in the frozen endpoint, and subsequently allocates calls around visible thermodynamic structure. The acquisition combines uncertainty, boundary proximity, numerical risk, and soft repulsion, but only returned BdG calculations enter the accepted dataset. The endpoint contains 7,434 exact observations: 1,867 Normal, 715 uniform-SC, and 4,852 FFLO labels. Its final phase-map change is 0.001629, the two boundary shifts are 0.004167 and 0, and trusted surprise is 0 for 127 trusted calls, satisfying the configured stopping rules. The separate 129-call hard-risk layer is a numerical frontier rather than a fourth phase or evidence that uncertainty has been eliminated. The exact-sample map supports an audited two-dimensional thermodynamic reconstruction under the registered criterion while retaining finite numerical risk.

[Inline FIG-02-THERMO; frozen caption unchanged.]



### Topology-resolved FFLO structure

Topology is evaluated only after thermodynamic acceptance and only on trusted gapped FFLO observations. A full-zone gap guard and Pfaffians P0 and Ppi determine the Z2 index, assigning cFFLO and tFFLO. Within the 4,345-point endpoint, 3,213 trusted gapped FFLO states divide into 2,290 cFFLO and 923 tFFLO observations and form a connected sampled structure. The contour and brackets diagnose a gap-closing transition; neither is an exact boundary, and the closing is not a stable third phase. The configured k=8 audit returns Decision A, with coverage p95=0.006185 below 0.00625 and the remaining registered metrics passing. For k=6, coverage p95=0.006269 lies slightly above the same threshold, so the configured pass retains a sensitivity and does not exhaust all refinements or nodal alternatives.

[Inline FIG-03-STAGEIII-TOPOLOGY; frozen caption unchanged.]



### Higher-dimensional exploration

The same oracle-scheduler separation extends the sampled coordinates to three dimensions over the finite mu interval [-0.5,1.5]. The endpoint contains 7,081 exact observations: 2,350 Normal, 141 uniform-SC, and 4,590 FFLO states. Trusted FFLO-only topology support comprises 3,145 cFFLO and 1,445 tFFLO points and remains discrete rather than a fitted surface. Four explicit fixed-mu bins centered at -0.45, 0.15, 0.85, and 1.45, each with half-width 0.025, expose exact slice-to-slice structure without interpolation. These point clouds and slices demonstrate higher-dimensional exploration that reveals thermodynamic and topology-aware structures across the sampled domain. The final strict audit exceeds the registered map-change, surface-shift, coverage, and surprise thresholds, while component counts vary 3 to 4 to 3. Thus the strict 3D topology convergence gates are not all closed; visual continuity is not a continuous surface, and the result is not a complete 3D atlas.

[Inline FIG-04-STAGEIV-3D-EXTENSION; frozen caption unchanged.]



## Discussion

The combined results establish a reliability-aware active-learning workflow in which exact BdG calculations retain label authority. The 2D audits show what closes under registered criteria, while the 3D extension demonstrates why selective exact sampling becomes more consequential as the domain grows. The 3D extension is bounded: discrete support reveals structure, but failed strict criteria prevent a converged-surface or complete-atlas claim. The matched Stage-V audit gives no overall winner: V3 has denser support, whereas V4 has lower sequential topology surprise. V4 does not lower the offline graph uncertainty, and none of its four candidate nodes has validated local same-identity support. Learned acquisition is an exploratory mechanism, not evidence of superiority or autonomous new-phase discovery. As a future direction, Stage VI indicates learnability of thermodynamics, q, and conditional Pfaffian quantities. The frozen export lacks fixed-k spectra, a DOS proxy, and validated spectral moments, and replay uncertainty is not uniformly monotonic. A physical world model must validate descriptors and uncertainty before production EIG can be authorized.
