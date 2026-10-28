# v8 verification summary

Date: 2026-09-12. Status: partial revision candidate; HUMAN_WAIT, not publication acceptance.

## Automated checks

- Existing four-node LangGraph profile `reader_revision_v8`: completed through audit, stopped at scientific-choice gate.
- Paired clean build: double-column main 6 pages, single-column review 8 pages, SM 15 pages.
- Shared source fingerprint: `ee2a8554ec7fc170c0ae996b45788e105b578d4379a8fbe820159ae545dc5724`.
- No overfull boxes, undefined references/citations, or multiply-defined references in the final three build logs.
- Document-workflow tests: 61 passed. These are not numerical-physics tests.
- Fig.2 and S2/S3 candidates: Figure Engine hard-fail checks pass; human gates remain waiting. Initial compact PNG export failed the minimum raster dimensions; export DPI was raised to 400 without changing vector geometry or relaxing the rule.
- Protected v7 files and frozen figure hashes match their input manifest; protected free-energy passage is byte-identical.
- R01/R12 use saved arrays only. Endpoint ordering/phase identity checked; full history and physical correctness are not inferred from this.

## Visual review and open defects

Rendered and inspected main pages 1–6, review pages 1–8 and SM pages 1–15. Re-inspected affected SM pages after the last wording change.

- New Fig.2 count no longer obscures the low-exchange strip in the main phase panel. Allocation shade and dashed references are explained in the caption. Some inherited base ticks remain small: the text-run diagnostic includes 6.8 pt at native width, approximately 6.3 pt in the single-column review's 164.5-mm insertion. Do not claim a universal 7-pt placed-size pass.
- SM algorithms are complete blocks; the pool definition, sampling equation and their surrounding text are no longer separated by floating method diagrams. S2/S3 retain original font sizes on shorter native canvases; SM pp.5–7 no longer have the large whitespace caused by those oversized method overviews.
- Main Fig.1, Fig.3, Fig.4 and S5 remain their existing assets, not newly certified artwork. Fig.3 axis/crowding, Fig.4 small bins/axis labels, and S5 small labels/legend clearance remain open under R08.
- Main-text floating figures still interrupt some cross-page sentence continuations. R09 is improved in the SM, not globally passed.
- R01/R02/R03/R11/R12/R13 author-dependent claims, figure choices and identity/access fields remain unresolved. The preserved wording in dependent passages is not an approval of it.

## Delivery

Three synchronized Markdown companions, seven SM source tables plus provenance/count/placement CSVs, important figure PNGs, page previews, decision log, literal source/wrapper diff and portable LaTeX source ZIP accompany the PDFs. Read README.md and AUTHOR_REVIEW.md before considering any manuscript gate.

No research code, data, thresholds or frozen scientific definitions changed; no BdG, training, topology solve, boundary extraction, new literature retrieval, commit, push or data publication was performed.
