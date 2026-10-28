# Figure 1 engine-v3 specific validation

- Status: **PASS**
- Hard failures: **0**

| Result | Check | Detail |
|---|---|---|
| pass | `engine_preflight` | Versioned Figure Engine preflight exists. |
| pass | `proposal_gate_recorded` | The separate v3 proposal gate is explicitly approved. |
| pass | `v1_frozen` | Frozen engine-v1 hashes match the benchmark. |
| pass | `v2_frozen` | Frozen engine-v2 files match its regeneration manifest. |
| pass | `primary_card_area` | Primary card fraction=0.6025; required >=0.55. |
| pass | `primary_scientific_area` | Landscape/central fraction=0.3784; required >=0.35. |
| pass | `declared_scientific_fill` | Scientific content/landscape fraction=0.7159; required >=0.65. |
| pass | `measured_real_surface_fill` | Measured embedded surface/landscape fraction=0.7169; required >=0.65. |
| pass | `primary_is_scientific_subobject` | Primary visual object resolves to the landscape scientific subobject. |
| pass | `topology_subobject_registered` | Topology is a separate registered scientific subobject. |
| pass | `workflow_route_count` | Found 7 main and 7 secondary workflow ribbons. |
| pass | `required_scientific_labels` | All required labels are present. |
| pass | `forbidden_claims_absent` | No forbidden/prohibited claim appears in artwork. |
| pass | `topology_scientific_separation` | Representative minima, gap/Pfaffian topology, and boundary-state semantics remain separate. |
| pass | `rendered_bbox_audit` | Browser-measured registered geometry passes. |
| pass | `exact_print_size` | PDF is exactly 178.00 × 120.15 mm. |
| pass | `print_previews` | 300-dpi color and grayscale previews exist. |
| pass | `artwork_word_budget` | Artwork word count=94; maximum=115. |
| pass | `no_physics_recomputation` | Asset provenance records physics_recomputed=false. |
| pass | `human_gate_held` | Automated validation leaves final Figure 1 approval to humans. |

Automated validation does not pass `GATE_FIG1_ENGINE_V3_FINAL_APPROVAL`.
