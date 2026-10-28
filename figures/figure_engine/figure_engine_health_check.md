# Figure engine health check

**Status:** `pass_with_known_fig1_failure`  
**Date:** 2026-08-31  
**Scope:** architecture and validation only; no manuscript artwork regenerated

## Executable architecture

- Canonical rule files loaded and hashed: 15/15
- Registered machine-enforced rules: 52
- Explicitly named human-review-only rules: 8
- Valid contracts: 6 (EXAMPLE-FIG1-SCHEMATIC, EXAMPLE-HIGH-DIMENSIONAL, EXAMPLE-PHASE-MAP, EXAMPLE-TOPOLOGY, FIG-01-PIPELINE-V5-ENGINE, FIG-01-ENGINE-V3-PROPOSAL)

## Tests

- Unit tests: **PASS** (28)
- Intentionally invalid validator fixture: **FAIL**
- Corrected control fixture: **PASS**
- Expected invalid conditions detected: 17
- Missing expected detections: 0

- Strict-data negative fixtures: **PASS** (15)

## Visual narrative and scientific-asset fill

- Parent-card substitution negative test: **DETECTED**
- Scientific-content fill-ratio negative/control check: **PASS**
- Frozen Fig.1-v1 surface fill ratio: **0.2029** (required proposal minimum 0.65)
- Expected frozen-v1 fill failure detected: **YES**

## Figure 1 regression

- Artwork status under expanded lint: **FAIL**
- Expected hard failures: `text_border_clearance`
- Actual hard failures: `text_border_clearance`
- Regression signature matches: **YES**

- Frozen SVG/PDF/PNG hashes unchanged: **YES**

- Frozen engine-v1 SVG/PDF/PNG hashes unchanged: **YES**

The Figure 1 failure is evidence that the stronger engine detects a previously missed
3.25-px text-to-card-border clearance against the 4-px hard minimum. The v5 artwork
remains read-only, and its human approval gate remains `HUMAN_WAIT`.

Automated engine health does not approve any manuscript figure.
