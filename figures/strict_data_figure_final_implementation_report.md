# Strict Data Figure Workflow final implementation report

## Outcome

The existing Figure Engine now has an executable strict-data track. Figure 2
has reached `GATE_FIG2_HUMAN_SCIENCE_REVIEW`; Figures 3 and 4 have contracts and
readiness reports only. Fig.1 engine-v4 is frozen and unchanged. No physics was
recomputed.

## Required implementation questions

1. **Existing components found.** The engine already provided a versioned
   constitution, contract/rule loaders, preflight hash records, panel assets,
   geometry/typography/color/scientific/visual-narrative validators, an
   enforcement registry, builders, templates, and negative-control health
   tests.

2. **Components reused.** Schema loading, canonical style tokens, Matplotlib
   rc derivation, preflight hashing, geometry manifests, editable-SVG checks,
   postflight lint aggregation, visual-narrative warnings, and the existing
   health-check harness are reused directly.

3. **Components extended.** Schema v3 adds `strict_data`; the engine now loads
   `strict_data_rules.yaml`, routes panels, checks artifact hashes/rows/fields/
   units, enforces transformation authorization, and runs strict-data negative
   controls. The approved iteration gradient was added as a versioned semantic
   palette scale rather than an ad-hoc artwork palette.

4. **Router logic.** Every panel declares one of six types:
   `EXACT_SAMPLE_DATA`, `DERIVED_NUMERICAL_DATA`,
   `DERIVED_BOUNDARY_GEOMETRY`, `INTERPOLATED_FIELD`,
   `SCHEMATIC_ANNOTATION`, or `HYBRID_DATA_SCHEMATIC`. Derived/interpolated
   routes add `GATE_TRANSFORMATION_AUTHORIZATION`; all routes traverse data
   scope, provenance, semantics, reproducibility, scientific invariants,
   narrative, style, export, human review, and freeze gates.

5. **Machine-enforced rules.** Required panel fields, source hashes/versions/
   rows/fields/units, row reconciliation, missing-data policy, phase mapping,
   phase counts, hard-risk exclusion from phase identity, color mapping,
   trusted topology masks, k=6 sensitivity declaration, gap-closing semantics,
   raw convergence representation, 3D non-convergence wording, transformation
   authorization, output existence, geometry, typography, palette, editable
   SVG, and export integrity are automated.

6. **Contract-plus-evidence rules.** Scientific roles, filters, trust masks,
   boundary algorithms/support definitions, numerical tolerances, allowed and
   forbidden claims, caveats, and exact-versus-derived encodings are declared
   in the contract and verified against registered artifacts.

7. **Human-only review.** First fixation, scientific emphasis, whether the
   exact-sample hero adequately communicates both thermodynamic adjacencies,
   caption adequacy, and final publication aesthetics remain human decisions.
   Saliency is warning-only and never certifies visual approval.

8. **Fig.2 panel evidence.** Panel a uses the real iter000 512-call selection,
   with 506 frozen Stage I exact labels and six explicitly unavailable results.
   Panel b uses all archived selections at iterations 0-34 (9216 calls).
   Panel c uses the 7434-row Stage II endpoint plus the separate 129-call
   hard-risk audit manifest. Panel d directly extracts archived raw stop metrics
   for all 35 iterations.

9. **Stage II counts.** Independently reproduced: 7434 total, 1867 Normal,
   715 uniform SC, and 4852 FFLO. Any mismatch would have stopped generation.

10. **Cold start/history.** Both are present. The cold start is the true
    `random_grid` iter000 selection. Stage I provides iterations 0-30 and the
    Stage II continuation provides 31-34; duplicated 26-30 histories are
    coordinate-identical and the canonical route uses Stage I for 0-30.

11. **Transformations.** No interpolation, smoothing, gridding, resampling,
    aggregation, boundary re-extraction, or label change occurred for Fig.2.
    Because final dense-monitor values and final boundary coordinates are
    absent, no new smooth thermodynamic boundary is drawn or claimed.

12. **Raw convergence metrics.** The main panel shows phase-map change,
    Normal-SC boundary shift, uniform-SC-FFLO boundary shift, and trusted
    surprise in native units. At iteration 34 they are 0.001628727653,
    0.004166666667, 0, and 0, respectively; configured thresholds are 0.002,
    0.004166666667, and 0.05 where applicable. Raw rebounds remain visible.

13. **Hard-risk separation.** Hard-risk is an open purple-ring overlay with a
    separate legend entry; Normal/uniform-SC/FFLO retain the only phase fills.
    The audit manifest has 129 archived exact calls; 19 coordinate matches
    remain in the final trusted endpoint. It is neither a fourth phase nor an
    assertion that all uncertainty was removed.

14. **Negative fixtures.** All 13 stably fail: wrong phase count, phase-color
    swap, hard-risk as fourth phase, missing units, unauthorized interpolation,
    unauthorized smoothing, missing topology trust mask, missing k=6
    sensitivity, gap closing as a stable phase, normalized-gate-ratio-only
    convergence, complete-3D claim, missing 3D non-convergence caveat, and
    indistinguishable exact/interpolated layers.

15. **Fig.3 readiness.** `CONTRACT_READY_BLOCKED_BY_FIG2_HUMAN_GATE`. Evidence
    contains 4345 rows and 3213 trusted gapped FFLO points: 2290 cFFLO and 923
    tFFLO. The k=8 coverage p95 is 0.006185; k=6 is 0.006269 and must be shown.
    Gap closing and local brackets remain diagnostics.

16. **Fig.4 readiness.** `CONTRACT_READY_WITH_CAVEAT_BLOCKED_BY_FIG2_HUMAN_GATE`.
    The endpoint has 7081 exact 3D rows with thermodynamic counts 2350/141/4590
    and topology counts 3265/1466. Exact clouds and 20 explicit fixed-$\mu$
    slices are contract-eligible. Continuous RBF/KNN surfaces remain blocked;
    strict 3D topology convergence is not closed.

17. **New-physics blocker.** None blocks the current Fig.2 review package.
    New calculations would be required to formally close the 3D topology result
    or replace missing dense/boundary evidence, but those actions are outside
    this task and remain unauthorized.

18. **Current gate.** Fig.1 is `SCIENTIFIC_FREEZE`. Fig.2 is
    `GATE_FIG2_HUMAN_SCIENCE_REVIEW = HUMAN_WAIT`. Fig.3 and Fig.4 rendering
    gates are blocked by the Fig.2 human gate. Global figure approval, final
    LaTeX integration, external retrieval, and new physics remain unpassed.

19. **Files for human review.** Review `fig2_strict_data.pdf`,
    `fig2_contact_sheet.pdf`, `fig2_scientific_validation.md`,
    `fig2_visual_narrative_audit.md`, `fig2_evidence_readiness.md`, and
    `figure_validation_report.md` in `figures/fig2_strict_data/`.

20. **Single-command rebuild.** From the repository root run:

    ```bash
    PYTHONPATH=. /opt/homebrew/bin/python3 figures/fig2_strict_data/reproduce_review_package.py
    ```

## Verification summary

- Figure Engine health: 15 rule bundles loaded; 52 enforcement rules; 28/28
  unit tests pass; 13/13 strict-data negative cases detected.
- Fig.2 generic lint: zero hard failures and zero warnings; first fixation is
  still explicitly human review.
- Fig.2-specific scientific/provenance validation: 21/21 pass.
- Deterministic rendering: two consecutive PNG hashes agree.
- PDF: one page, 504.567 x 340.157 pt, matching the 178 x 120 mm contract
  within PDF point rounding; the contact sheet is one US-Letter page.

```yaml
GATE_FIG2_HUMAN_SCIENCE_REVIEW:
  READY
```
