# Full-graph implementation plan

## Implemented in this slice

- isolated directory and preserved user prompt;
- curated artifact/provenance and readiness audits;
- scope, claim, evidence, and figure registries;
- executable graph skeleton with offline compatibility runner;
- explicit node contracts and human gates;
- Stage III Results-by-figure approval package and isolated LaTeX fragment;
- validation tests for sequencing, registries, gates, and evidence trace.

Contract revalidation now covers the prompt-required claim fields, the five
staged stopping boundaries, and the observation/interpretation/source/figure
metadata required for every Results paragraph.

## Phase 2 — after scope and vertical-slice approval

1. Recover Stage II/III frozen raw bundles and regenerate the manifest.
2. Replace report-level Stage III evidence with raw tables and final figure
   provenance.
3. Implement a real figure-contract loader and sentence-level evidence-trace
   store.
4. Expand Results-by-figure to Figures 1–3; keep Figure 4 conditional.
5. Implement Methods from canonical model/numerics definitions.
6. Adapt the upstream citation retrievers behind an explicit network gate.
7. Add cross-section number/notation/claim consistency checks.
8. Add isolated RevTeX build, page/word/figure checks, and visual PDF QA.

## Phase 3 — requires a scientific decision and possibly new computation

Choose between a physics-first paper and a method-efficiency paper. The latter
requires matched-budget baselines, multiple seeds, and a hidden evaluator.
No such computation is authorized by this architecture task.
