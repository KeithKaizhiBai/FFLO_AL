# Visual narrative report

- Hard-rule status: **PASS**
- Hard failures: **0**
- Warnings: **0**
- Human-review judgments: **1**
- Primary visual object: `exact_3d_thermodynamic_cloud`
- Declared reading order: `exact_3d_thermodynamic_cloud -> exact_3d_topology_support -> fixed_mu_slice_audit`

## Warning-oriented saliency proxy

1. `exact_3d_thermodynamic_cloud`
2. `exact_3d_topology_support`
3. `fixed_mu_slice_audit`

| Result | Severity | Check | Detail |
|---|---|---|---|
| pass | hard_fail | `narrative_primary_scientific_subobject` | The primary visual object resolves to a registered scientific subobject, never its parent card. |
| pass | hard_fail | `narrative_primary_area` | The primary scientific subobject satisfies its explicit contract area requirement. |
| pass | hard_fail | `narrative_reading_order` | The declared reading order begins with the primary scientific subobject and all narrative regions resolve. |
| pass | hard_fail | `scientific_asset_fill_ratio` | Declared scientific content fills the asset bbox above its figure-specific minimum. |
| pass | warning | `narrative_saliency_rank` | Blurred saliency is a warning-only proxy; it never certifies first fixation or aesthetic quality. |
| pass | warning | `narrative_contrast_hierarchy` | Primary mean contrast should not fall below the median declared tertiary-region contrast. |
| pass | warning | `narrative_equal_weight_regions` | Four or more regions with near-primary mass trigger an equal-weight warning. |
| review/fail | human_review | `narrative_first_fixation_human_review` | Human review is required to decide whether actual first fixation matches the declared reading order. |

Saliency and first fixation are not scientific criteria and are not automatic aesthetic approval.
Automated validation does not pass a human proposal or figure gate.
