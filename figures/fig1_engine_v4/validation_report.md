# FIG-01-ENGINE-V4 validation report

- Status: **PASS**
- Hard failures: **0**
- Warnings: **0**
- Human-review judgments: **1**
- Checks passed: **51/52**

| Result | Severity | Category | Check | Detail |
|---|---|---|---|---|
| pass | hard_fail | output | `editable_svg_present` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/figures/fig1_engine_v4/fig1_engine_v4.svg |
| pass | hard_fail | output | `pdf_present` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/figures/fig1_engine_v4/fig1_engine_v4.pdf |
| pass | hard_fail | output | `png_preview_present` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/figures/fig1_engine_v4/fig1_engine_v4.png |
| pass | hard_fail | output | `png_resolution` | PNG preview is 2103x1420 px. |
| pass | hard_fail | output | `white_background` | Top-right corner RGB=(255, 255, 255). |
| pass | hard_fail | output | `pdf_single_page` | Pages:           1 |
| pass | hard_fail | output | `pdf_page_size_recorded` | Page size:       504.567 x 340.583 pts |
| pass | hard_fail | integration | `engine_preflight` | Engine preflight record matches the current contract and all canonical rule hashes: /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/figures/fig1_engine_v4/engine_preflight.json |
| pass | hard_fail | asset | `declared_asset_semantics` | Every visual-role object has a well-formed asset declaration. |
| pass | hard_fail | asset | `schematic_not_quantitative_evidence` | Quantitative scientific asset types are not supplied by schematic generators. |
| pass | hard_fail | asset | `required_scientific_features` | Every typed scientific asset declares the features required by asset_rules.yaml. |
| pass | hard_fail | svg | `parse` | SVG parses successfully. |
| pass | hard_fail | svg | `editable_text` | SVG contains 76 editable text nodes. |
| pass | hard_fail | svg | `raster_policy` | SVG contains 1 image nodes; raster exception declared=True. |
| pass | hard_fail | typography | `svg_nonuniform_scaling` | No SVG group uses non-uniform scale. |
| pass | hard_fail | typography | `svg_skew` | No SVG group uses skew transforms. |
| pass | hard_fail | svg | `engine_metadata` | SVG records figure id, contract path, and engine schema. |
| pass | hard_fail | color | `semantic_mappings` | Every contract color role maps to a canonical semantic family. |
| pass | warning | color | `approved_palette` | Unknown SVG colors: 261; strict palette=False. |
| pass | hard_fail | color | `minimum_contrast` | Registered contrast pairs: 3; failures: 0. |
| pass | hard_fail | scientific | `required_labels` | All contract-required scientific labels are present. |
| pass | hard_fail | scientific | `prohibited_claims` | No contract-prohibited claim appears in the artwork. |
| pass | hard_fail | geometry | `geometry_manifest` | /Users/baikaizhi/Desktop/PhD_study/4th/GBU_SC/AL_FFLO/figures/fig1_engine_v4/geometry_manifest.json |
| pass | hard_fail | geometry | `canvas_clipping` | All registered objects fit the canvas. |
| pass | hard_fail | geometry | `parent_containment` | All registered child objects fit their parent boxes. |
| pass | hard_fail | geometry | `independent_card_gutters` | Independent cards retain at least 20 px clearance. |
| pass | hard_fail | layout | `primary_visual_area` | Primary registered area fraction 0.6025; required minimum 0.5500. |
| pass | hard_fail | layout | `declared_area_requirements` | All contract-declared scientific subobjects meet their minimum area fractions. |
| pass | hard_fail | geometry | `text_text_overlap` | No undeclared text-text overlaps. |
| pass | hard_fail | geometry | `text_border_clearance` | Registered text clears its parent border by at least 4 px. |
| pass | hard_fail | geometry | `text_plot_clearance` | Text clears unrelated plots by at least 6 px unless an overlay is declared. |
| pass | hard_fail | geometry | `phase_label_clearance` | Phase-label clearance is at least 10 px. |
| pass | hard_fail | geometry | `leader_text_clearance` | Annotation leaders clear unrelated text by at least 5 px. |
| pass | hard_fail | geometry | `leader_leader_crossing` | Annotation leaders do not cross one another. |
| pass | hard_fail | typography | `minimum_font_size` | All registered text is at least 7 pt. |
| pass | hard_fail | typography | `normal_width_fonts` | No condensed/narrow/compressed fonts are used. |
| pass | hard_fail | typography | `nonuniform_text_scaling` | No text uses non-uniform scaling. |
| pass | hard_fail | typography | `skewed_text` | No text uses skew transforms. |
| pass | warning | typography | `preferred_math_font` | Preferred STIX math stack is used for new output; legacy fallbacks are reported. |
| pass | warning | typography | `sans_font_stack` | Non-math text uses the approved normal-width sans stack. |
| pass | hard_fail | arrow | `workflow_source_target` | Every main workflow route has distinct registered source and target roles. |
| pass | hard_fail | arrow | `workflow_profile` | Main workflow routes inherit tapered width, opacity, and sampling tokens from arrow_rules.yaml. |
| pass | hard_fail | arrow | `workflow_card_crossing` | Main workflow route interiors do not pass through unrelated cards. |
| pass | hard_fail | arrow | `annotation_length` | Annotation leaders do not exceed 48 px. |
| pass | hard_fail | visual_narrative | `narrative_primary_scientific_subobject` | The primary visual object resolves to a registered scientific subobject, never its parent card. |
| pass | hard_fail | visual_narrative | `narrative_primary_area` | The primary scientific subobject satisfies its explicit contract area requirement. |
| pass | hard_fail | visual_narrative | `narrative_reading_order` | The declared reading order begins with the primary scientific subobject and all narrative regions resolve. |
| pass | hard_fail | visual_narrative | `scientific_asset_fill_ratio` | Declared scientific content fills the asset bbox above its figure-specific minimum. |
| pass | warning | visual_narrative | `narrative_saliency_rank` | Blurred saliency is a warning-only proxy; it never certifies first fixation or aesthetic quality. |
| pass | warning | visual_narrative | `narrative_contrast_hierarchy` | Primary mean contrast should not fall below the median declared tertiary-region contrast. |
| pass | warning | visual_narrative | `narrative_equal_weight_regions` | Four or more regions with near-primary mass trigger an equal-weight warning. |
| fail | human_review | visual_narrative | `narrative_first_fixation_human_review` | Human review is required to decide whether actual first fixation matches the declared reading order. |

Automated validation does not pass the human approval gate.
