# Evidence-first PRL LangGraph Q&A

Date: 2026-08-07

## Question

How should the mature Active Learning FFLO results be converted into a PRL
without promoting incomplete Stage IV--VI evidence or drafting prose before
the claims are frozen?

## Answer

Use an evidence-first graph whose scientific generation order differs from
article reading order. Audit provenance and readiness first, freeze scope and
claim wording, draft Results figure by figure with paragraph-level evidence
metadata, then draft Methods and Discussion. Run Introduction citation retrieval
only after Results claims are frozen; derive Abstract and Title last; finish with
consistency, citation, length, LaTeX, and human-approval gates.

## Recommended scope

Stages I--III are the current defensible central line. Stage III is the strongest
result because its configured topology audit returned Decision A. The Results
must also report the tight sensitivity margin: k=8 coverage p95 is 0.006185
against 0.00625, while k=6 gives 0.006269. Stage IV is a non-converged extension;
V3/V4 superiority is not established; Stage VI is not EIG-ready.

## Evidence gap

The local checkout lacks the Stage II/III raw endpoints and the Stage III final
convergence/figure bundle. It also lacks matched-budget baselines, multiple
seeds, and hidden ground truth. Therefore the Stage III vertical slice can be
evidence-traceable at the audit-record level but cannot yet produce the final
publication figure.

## Output

```text
manuscript_langgraph/
output/pdf/manuscript_langgraph_implementation/
output/pdf/manuscript_langgraph_stageiii_vertical_slice/
```
