# Manuscript LangGraph contract revalidation

Date: 2026-08-07

## Question

Does the stored publication-oriented prompt correspond to an executable,
evidence-first manuscript workflow whose registries, paragraph traces, tests,
and human gates agree with the prompt?

## Answer

Yes, after repairing interface drift found during revalidation. The prompt was
already preserved verbatim except for a terminal newline. The claim registry
now uses the prompt-required `scientific_status` and `evidence_ids` fields. Each
Stage III paragraph trace now contains an explicit observation,
interpretation, evidence source, figure reference, caveats, and unsupported-
sentence flags.

The 15-node graph stops at `GATE_SCOPE_APPROVAL` by default. Approved staged
invocations independently reach `GATE_CLAIM_APPROVAL`,
`GATE_FIGURE_APPROVAL`, the Introduction retrieval gate, and
`GATE_FINAL_MANUSCRIPT_APPROVAL`. Fourteen tests and the staged workflow
validator pass.

## Scientific status

The recommended scope remains Stages I--III with Stage III central. Stage IV
is conditional and not formally converged; Stage V is Discussion-only with no
V3/V4 superiority claim; Stage VI remains Outlook. Three publication-critical
Stage II/III artifacts remain missing locally, so full manuscript expansion and
final Figure 2/3 reproduction are not authorized.

## Outputs

```text
manuscript_langgraph/prompts/publication_oriented_manuscript_langgraph_prompt.md
manuscript_langgraph/audits/manuscript_artifact_manifest.csv
manuscript_langgraph/manuscript_langgraph_implementation_report.md
manuscript_langgraph/manuscript_langgraph_implementation_report.pdf
output/pdf/manuscript_langgraph_implementation/
```
