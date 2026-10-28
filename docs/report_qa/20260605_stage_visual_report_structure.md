# Stage-Ordered Visual Report Structure

Date: 2026-06-05

## Question

How should the second major-stage report present representative figures from
each calculation stage, especially phase diagrams, while explaining which
improvements produced which visual changes?

## Answer

Use a compact stage-by-stage visual narrative rather than a loose figure
gallery.

Recommended order:

1. Exact warm-up baseline: show the original exact phase diagram as the
   physical reference.
2. Active-learning boundary refinement: show the boundary-focused exact points
   and combined eta phase map to explain that ML concentrated exact calls near
   boundaries without replacing the BdG oracle.
3. Label-closed active-learning benchmark: show the later final exact eta map
   with revised boundaries, selected points by source, and selection-focus
   curve to demonstrate that the corrected oracle still supports boundary
   discovery.
4. ML/acquisition profile benchmark: show the ML training architecture and
   full-vs-simple-phase selected-by-boundary-type plots.  In particular, the
   mini-report Fig. 11 simple-phase panel is useful because it shows strong
   normal/SC focusing while also exposing weaker uniform-SC/FFLO coverage.
5. Response numerical audit: show response q-window and q-density figures to
   explain why unstable positive eta claims were downgraded.
6. Phase q-window and Delta audit: show qopt shifts, phase-change map, DeltaF
   refinement, and local-minimum branch candidates to explain that expanded
   q-window coverage moved high-JA phase labels while Delta refinement acted
   mainly as a guardrail.
7. Local-refinement refactor: show the stage-gate figure and table, but do not
   claim a new post-refactor phase diagram until the Stage 2-4 GPU variant-suite
   return archive has been imported and passed.

The appendix should retain the active-learning workflow diagram and use it to
connect implementation detours to the exact-data return path: package-local
imports, RUN_ROOT output policy, CUDA probes, explicit PACKAGE_ROOT export, and
return-archive checks.

## Caveat

The current local-refinement stage remains pending external GPU evidence.  Any
report figure from earlier stages is baseline or audit context, not a
post-refactor physics result.
