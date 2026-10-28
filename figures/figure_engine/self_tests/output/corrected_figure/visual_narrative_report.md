# Visual narrative report

- Hard-rule status: **PASS**
- Hard failures: **0**
- Warnings: **1**
- Human-review judgments: **1**
- Primary visual object: `landscape`
- Declared reading order: `landscape -> supporting_evidence`

| Result | Severity | Check | Detail |
|---|---|---|---|
| pass | hard_fail | `narrative_primary_scientific_subobject` | The primary visual object resolves to a registered scientific subobject, never its parent card. |
| pass | hard_fail | `narrative_primary_area` | The primary scientific subobject satisfies its explicit contract area requirement. |
| pass | hard_fail | `narrative_reading_order` | The declared reading order begins with the primary scientific subobject and all narrative regions resolve. |
| pass | hard_fail | `scientific_asset_fill_ratio` | Declared scientific content fills the asset bbox above its figure-specific minimum. |
| review/fail | warning | `narrative_saliency_evidence` | No final PNG was supplied; saliency remains unevaluated and cannot be reported as an automatic pass. |
| review/fail | human_review | `narrative_first_fixation_human_review` | Human review is required to decide whether actual first fixation matches the declared reading order. |

Saliency and first fixation are not scientific criteria and are not automatic aesthetic approval.
Automated validation does not pass a human proposal or figure gate.
