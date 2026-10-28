# Manuscript worklog

## 2026-08-27

- Recorded explicit human approval of the Results/caption review package and resumed the same `active_learning_prl_v1` SQLite thread.
- Replaced the Methods, Supplement, and Discussion/Outlook gate stubs with evidence-linked builders; generated LaTeX/Markdown drafts, trace ledgers, audits, and two intermediate review PDFs without new physics or external retrieval.
- Added pre-retrieval Introduction claim/citation requirement outlines, but performed no external search.
- During full rendered-PDF inspection, detected that approved final Fig. 4 shows three fixed-$\mu$ slices at -0.35, 0.55, and 1.25 while the approved Results/caption text describes four slices from the earlier strict-data prototype.
- Added `GATE_RESULTS_FIG4_CAPTION_CORRECTION`; stopped at `HUMAN_WAIT` with a two-fragment correction proposal. The generated PDFs are layout previews until the correction is approved and rebuilt.
- Manuscript workflow tests: 19/19 passed. Main preview: 7 pages; SM preview: 2 pages; all pages visually inspected with no clipping or overlap.

- Resumed the real `active_learning_prl_v1` SQLite thread after explicit Figure approval.
- Added one graph-native Results package builder and one independent human-review gate.
- Generated Results, captions, sentence-level traces, numerical/semantic audits, conflict ledger, and a seven-page review PDF from frozen evidence.
- Independently verified Stage II--IV counts, topology masks, thresholds, $k$ sensitivity, $\mu$ range, and all frozen Figure 1--4 hashes.
- Ran 18/18 manuscript workflow tests and visually inspected all seven review-PDF pages.
- Stopped at `GATE_RESULTS_HUMAN_REVIEW=HUMAN_WAIT`; Results are not frozen and Methods/external retrieval were not executed.

## 2026-08-13

- Accepted and indexed the read-only Windows scientific evidence archive.
- Resolved canonical Stage II/III/IV endpoints without a conflicting-run gate.
- Recorded claim approval with modifications.
- Added evidence entries for archive integrity, cold-start history, hard-risk,
  fixed-mu slice audit, and curve-extraction repair.
- Frozen four candidate figure contracts and a section/evidence outline.
- Began deterministic figure regeneration; no physics calculation or external
  retrieval was run.
- Regenerated Figures 1--4 as PDF/SVG/PNG candidates and built Supplementary
  Figure S1 from frozen/derived audit tables.
- Completed 29 scientific/format checks and full-resolution visual QA.
- Built and rendered the six-page figure-approval package.
- Installed the project-declared LangGraph runtime plus SQLite checkpoint
  backend and invoked the registered 15-node graph.
- Saved the persistent `active_learning_prl_v1` checkpoint and stopped at
  `GATE_FIGURE_APPROVAL=HUMAN_WAIT`.
- Rebuilt the artifact manifest: 65 present, zero missing.
## 2026-08-27 — Approved Fig. 4 amendment and journal positioning

- Corrected the frozen Results paragraph/caption to the three fixed-$\mu$
  slices present in approved Figure 4: $-0.35$, $0.55$, and $1.25$.
- Rebuilt Results, main review, and Supplemental Material PDFs without changing
  artwork or physics.
- Refroze Results hashes and passed the Figure 4 correction gate.
- Verified 20/20 manuscript tests and 17-node/18-gate workflow validation.
- Rebound the Results review packet from the preserved four-slice strict-data
  prototype to the already approved final three-slice Figure 4; no artwork was
  regenerated.
- Current stop is `INTRO_CITATION_LOOP_GATE=HUMAN_WAIT`; external retrieval is
  false.
- Historical journal-policy audit (subsequently superseded by the user's format
  clarification): the architecture was assessed as Nature-Article-like and
  *Communications Physics* as a possible fit.

## 2026-08-27 — PRL style retained without a journal commitment

- Recorded `target_journal=undecided` after the user clarified that PRL is a
  visual-format preference only.
- Retained the REVTeX PRL-style main text and Supplemental Material without
  compressing the frozen Figures 1--4 narrative.
- Superseded the earlier venue recommendation in active workflow/handoff state.
- No physics, figure, claim, or retrieval gate changed.

## 2026-08-28 — Framework-figure source repair and LangGraph rewrite

- Audited the Results and main-manuscript compilers and found a hard-coded
  dependency on the legacy `manuscript_figures/final` main-figure bundle.
- Established a single canonical source map: Fig. 1 engine-v4; Figs. 2--4
  strict-data scientific freezes.
- Restored Fig. 4 Results/caption wording to the approved four fixed-$\mu$
  bins at $-0.45$, $0.15$, $0.85$, and $1.45$ (half-width 0.025).
- Added precompile and staged-copy SHA-256 enforcement, a manuscript figure
  source audit, CSV provenance table, LaTeX recorder check, and two regression
  tests. Legacy candidates remain preserved but are not authoritative.
- Invoked the 17-node Manuscript LangGraph to rewrite Results, Methods, SM,
  Discussion, and Outlook, rebuilding all three review PDFs without external
  retrieval or new physics.
- Verified 22/22 direct manuscript tests, workflow validation, all four exact
  embedded-figure hashes, and all 16 rendered PDF pages.
- Current stop remains `INTRO_CITATION_LOOP_GATE=HUMAN_WAIT`.

## 2026-08-28 — PRL-integrated main-text profile

- Added the versioned `PRL_INTEGRATED_MAIN_TEXT` publication profile and a
  separate `active_learning_prl_integrated_v1` graph/run branch.
- Reused the parent provenance/readiness, claim, figure, evidence, and
  canonical-asset authorities; the existing graph and intermediate PDF remain
  unchanged.
- Split Methods into seven evidence-traced packets and wove them into the
  corresponding scientific result paragraphs.
- Built a continuous provisional Introduction -> integrated Results/Methods ->
  Discussion candidate with no standalone Results, Methods, or Outlook
  sections.
- Inserted frozen Figures 1--4 inline with frozen captions and verified all
  source/staged hashes plus compiled placement.
- Preserved the Stage III $k=6$ sensitivity and all Stage IV finite-domain,
  discrete-slice, nonclosure, and no-surface caveats.
- Produced semantic, structure, method, figure, claim, split, length, compile,
  citation-gap, and caption-condensation review artifacts.
- Compiled a six-page candidate. The internal length audit reports
  `OVERLENGTH_CANDIDATE`; caption proposals are not applied.
- No external retrieval or physics recomputation occurred. Current stop:
  `GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW=HUMAN_WAIT`.

## 2026-08-28 — Fig. 2--4 single-column trial

- Added an isolated single-column trial builder and four regression tests.
- Kept Fig. 1 as a double-column float and converted Figs. 2--4 to ordinary
  full-column floats without editing their frozen PDFs, captions, or science.
- Compiled a four-page candidate with 0 overfull and 0 underfull boxes; all
  fonts are embedded and 31/31 manuscript tests pass.
- Verified source-candidate and frozen-figure hashes before and after the
  build. The parent gate remains `HUMAN_WAIT`.
- Full-page 200 dpi and figure-crop 300 dpi review finds the reading flow much
  improved, but the 0.4875 scaling projects 7 pt figure text to 3.41 pt.
- Recorded `PASS_AS_LAYOUT_PROOF_FAIL_AS_FINAL_DIRECT_SCALE`; production use
  requires native 86.8 mm Figure Engine variants of Figs. 2--4.

## 2026-08-28 — Verified citations and physics-first Introduction

- Recorded explicit authorization for external Introduction retrieval while
  keeping new physics, Results edits, and figure edits out of scope.
- Added `PRL_INTEGRATED_CITED_MAIN_TEXT` and child run
  `active_learning_prl_integrated_cited_v1`.
- Verified 12 sources and mapped them to six bounded FFLO, altermagnetism,
  topology, and active-learning background claims.
- Generated the registry, claim-to-citation matrix, BibTeX, retrieval audit,
  five-paragraph Introduction, and citation coverage audit.
- Compiled a six-page candidate with 12/12 references, no overfull boxes,
  embedded fonts, and unchanged frozen-figure hashes.
- Reordered only cited-branch float declarations after visual QA exposed a
  nearly blank page and a post-bibliography Fig. 4 placement.
- Passed 36/36 tests and legacy 17-node/18-gate workflow validation; inspected
  every final PDF page.
- Stopped at
  `GATE_CITED_INTRODUCTION_MANUSCRIPT_HUMAN_REVIEW=HUMAN_WAIT`.

## 2026-08-28 — Expanded citation policy and 50-reference candidate

- Added `PRL_INTEGRATED_CITED_MAIN_TEXT_V2` and child run
  `active_learning_prl_integrated_cited_v2` without modifying cited_v1.
- Replaced the 12-reference ceiling with a validator-enforced 45--55 range and
  target of 50; retained 12 T1 sources and admitted 38 contextual T2 sources
  through verified Crossref DOI metadata.
- Expanded the Introduction to six physics-first paragraphs and the
  claim-to-citation matrix to 12 bounded clusters spanning FFLO platforms,
  altermagnetic superconductivity, finite-$q$/class-D topology, and active
  learning/scientific ML.
- Verified 50 unique citation keys, 49 unique DOI records plus the manually
  cross-checked LO source, 50/50 cited/defined entries, and zero bibliography
  padding.
- Added Unicode normalization to metadata retrieval and an every-page-content
  compile guard.
- Corrected a nearly blank-page regression by reordering only the unchanged
  Fig. 2 float declaration in the v2 branch.
- Compiled a seven-page candidate with no undefined citations, no overfull
  boxes, embedded fonts, preserved caveats, and unchanged figure hashes.
- Passed 37/37 tests and inspected all seven rendered pages.
- Current stop remains
  `GATE_CITED_INTRODUCTION_MANUSCRIPT_HUMAN_REVIEW=HUMAN_WAIT`.
