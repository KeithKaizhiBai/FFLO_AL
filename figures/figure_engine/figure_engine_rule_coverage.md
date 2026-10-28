# Figure engine rule coverage

**Date:** 2026-08-31  
**Coverage policy:** a check is marked automatic only when a builder or lint
can return nonzero on a concrete violation.

## Historical Figure 1 failures

| Historical problem | Canonical rule | Enforcement | Negative/control evidence | Coverage |
|---|---|---|---|---|
| Font below print readability | typography minimum size | SVG builder + font lint | 5-pt fixture rejected; corrected 7-pt fixture passes | automatic |
| Condensed or stretched text | normal-width font; no non-uniform scale | font lint | Arial Narrow and 0.75x scale both detected | automatic |
| Text-text overlap | text geometry hard rule | collision lint | overlapping bboxes detected | automatic |
| Text too close to card border | 4-px border clearance | collision lint | fixture detected; Fig.1 regression found 3.25 px | automatic |
| Text obscures a scientific plot | 6-px plot clearance | collision lint | fixture detected | automatic with declared overlays |
| Cards touch or overlap | 20-px independent-card gap | bbox lint | overlapping fixture cards detected | automatic |
| Central physics too small | primary area threshold | bbox lint | 23.8% fixture primary fails 55% contract floor | automatic |
| Scientific subobject tiny inside a large container | declared internal area fraction | bbox lint | tiny landscape fixture detected | automatic when contract declares threshold |
| Parent card substituted for primary scientific object | visual-narrative primary resolution | contract + narrative lint | schema-v2 invalid fixture declares `primary_card`; corrected control declares `landscape` | automatic hard fail |
| Large landscape bbox but tiny rendered surface | figure-specific scientific-content fill ratio | narrative lint + frozen Fig.1 audit | invalid 2x2 content fails; corrected 160x80 content passes; frozen v1 ratio is 0.2029 | automatic hard fail when enabled |
| Unexpected reading-order saliency | blurred region-level visual mass | narrative lint | ranked and reported without failing the build | warning only |
| Actual first fixation | explicit visual-narrative human gate | narrative report | always retained as human review | human only |
| Arbitrary/unapproved color | semantic palette | SVG builder + strict color lint | `#00FF00` rejected/detected | automatic |
| Insufficient color contrast | registered contrast pairs | color lint | passing and failing contrast unit tests | automatic for registered pairs |
| Flat or visually dull color balance | low-saturation/background hierarchy | semantic palette + human review | no honest generic pixel metric | partial/human |
| Thin constant-width workflow arrows | tapered 6/17/23 profile | route lint | constant 1/1/1 fixture detected | automatic |
| Arrow passes through an unrelated card | route-interior rule | route lint | obstacle-card fixture detected | automatic for registered samples |
| Decorative arrow without meaningful endpoints | source/target roles | route lint | missing/distinct endpoints are hard failures | automatic for registration; intent human |
| Leader crosses another leader | annotation rule | collision lint | crossing fixture detected | automatic |
| Leader is excessively long | 48-px hard maximum | route lint | 156.2-px fixture detected | automatic |
| Primary state space loses to equal panel weight | hierarchy/no-equal-weight | contract + bbox lint | current Fig.1 central fraction 60.25% passes 55% floor | automatic for area; first fixation human |
| ML appears to replace exact physics | physics-first/prohibited claim | scientific text lint | forbidden phrase fixture detected | automatic for text; visual implication human |
| cFFLO/tFFLO transition conflated | topology asset features | asset contract + Fig.1-specific audit | requires gapped cFFLO, separate closing, reopened gapped tFFLO | contract + figure-specific + human |
| Boundary state mistaken for bulk band | boundary-state separation feature and prohibited claim | asset contract + scientific/figure-specific audit | required feature registered in Fig.1 contract | contract + figure-specific + human |
| Schematic used as quantitative evidence | asset provenance rule | asset lint | quantitative types marked schematic hard-fail | automatic for declared type/source |
| Required scientific label missing | contract label list | SVG text lint | unit test and full lint | automatic |
| Prohibited manuscript claim present | forbidden claim list | SVG text lint | synthetic forbidden claim detected | automatic |
| Builder ignores canonical files | mandatory preflight and exact current hashes | preflight + postflight + health check | 13/13 files loaded; deliberately stale hash rejected in unit test | automatic for reading/currentness |
| Builder reads rules but uses local constants | token inheritance | engine-native SVG/Matplotlib builders | native builders constrained; legacy v5 adapter remains local | partial/known debt |

## Validator self-test result

- Intentionally invalid fixture: **FAIL**, as required.
- Expected hard failures detected: **15/15**.
- Corrected control using the same framework: **PASS**.
- Missing expected detections: **0**.

Detailed reports:

- `self_tests/output/invalid_figure/figure_validation_report.md`
- `self_tests/output/corrected_figure/figure_validation_report.md`
- `self_tests/output/validator_self_test_summary.md`

## Fig.1 regression result

The unchanged v5 artwork is no longer treated as an unconditional generic-lint
pass. Under the strengthened engine it passes 42 of 45 checks but has:

- one hard failure: `text_border_clearance`;
- two declared legacy warnings: missing engine metadata and Times math fallback.

The hard failure is `reliability-row-1`, whose measured right clearance is
3.25 px against the 4-px minimum. The engine therefore caught a concrete issue
not reported by the older 12-category Fig.1 lint.

This is a successful regression test of the engine, not permission to change
the figure.

## Honest boundary of coverage

The engine can enforce geometry, registered semantics, provenance, and textual
claims. It cannot generically infer the truth of a topological invariant or the
perceptual quality of a 3D rendering from pixels. Those remain bounded by
frozen data, figure-specific audits, and explicit human review.

## Stage-IV diagnostic-threshold scope repair

The engine now distinguishes raw numerical stability diagnostics from physical
convergence criteria.

| Risk | Canonical rule | Enforcement | Negative/control evidence | Coverage |
|---|---|---|---|---|
| Operational cutoff drawn as a physical scale | cutoff line requires a declared scientific or numerical basis | strict-data panel router + figure-specific SVG audit | diagnostic-only fixture with a visible cutoff is rejected | automatic for declared diagnostic mode |
| PASS/FAIL inferred from an unsupported cutoff | diagnostic-only mode forbids pass/fail labels | strict-data panel router + prohibited-text audit | pass/fail negative fixture is rejected | automatic for declared diagnostic mode |
| High-dimensional scope hidden behind a score | finite-domain/discrete-support evidence caveat required | strict-data panel router | missing-scope fixture is rejected | automatic contract check |
| Equal small diagnostic panels rejected as a dashboard | Supplement-diagnostics layout profile | layout contract + geometry lint | six-panel Stage-IV SM contract passes | automatic geometry; hierarchy remains human-reviewed |

The complete health check passes 28 unit tests and 15/15 strict-data negative
fixtures. Automated passes do not approve the new Figure 4 or Supplemental
figure.
