# Decision Log: Numerical Reliability Audit

Date: 2026-05-23

## Scope

This report summarizes audit-only calculations following the discovery-mode
active-learning run `active_boundary_discovery_512seed_256x50`. The audits do
not modify the active-learning dataset, acquisition function, neural-network
surrogate, StopController, or production BdG oracle.

## Main Conclusions

- The discovery active-learning workflow recovered the main thermodynamic
  normal/SC and uniform-SC/FFLO boundary structure from random exact seeds.
- The free-energy phase rule remains valid within the scanned \((\Delta,q)\)
  window: compare the best positive-\(\Delta\) superconducting state with the
  normal state, then use \(q_{\rm opt}\) to distinguish uniform SC from FFLO.
- The tested high-\(J_A\), positive-\(\eta\) points are not robust diode-response
  evidence. Most become non-positive, change sign, or become unstable under
  q-window and q-density audits.
- The residual \(\eta=1\) cases at points 21 and 25 are response-pathology
  cases: one critical-current branch is zero or near zero, making \(\eta\)
  ill-conditioned.
- The remaining dominant uncertainty is not the \(\eta\) response. It is whether
  high-\(J_A\), low-\(T\) free-energy scans cover all relevant FFLO branches.

## Decisions

- Treat response-level positive \(\eta\) claims near high \(J_A\) as untrusted
  unless both q-window validity and q-density/branch-extremum stability are
  satisfied.
- Do not interpret \(\eta=1\) from \(I_c^-=0\) as robust positive diode physics.
  Mark it as ill-conditioned / branch-near-zero.
- Keep the current active-learning dataset unchanged. The audit results are
  diagnostic evidence, not new training labels.
- The next numerical audit should target the thermodynamic phase boundary, not
  only the response: expand the free-energy q-window, save \(F_{\min}(q)\), and
  inspect multiple low-energy local minima.

## Unresolved Issues

- A larger q-window may reveal a lower-energy FFLO branch outside the original
  high-\(J_A\), low-\(T\) scan window.
- Several high-\(J_A\) normal/SC boundary points remain close to the normal-state
  energy within Delta-refinement tolerance.
- Old cFFLO/tFFLO curves are reference curves only. The current oracle has not
  assigned pointwise topological labels.

## Recommended Next Calculation

Run a phase-boundary q-window and branch-minimum audit for selected high-\(J_A\),
low-\(T\), normal/SC-kink points:

1. expand the q-window for the free-energy scan;
2. save \(F_{\min}(q)=\min_\Delta F(\Delta,q)\);
3. extract multiple low-energy local minima;
4. compare old and expanded-window phase labels, \(q_{\rm opt}\),
   \(\Delta_{\rm opt}\), and \(\Delta F\);
5. check whether a lower FFLO branch appears outside the old q-window.
