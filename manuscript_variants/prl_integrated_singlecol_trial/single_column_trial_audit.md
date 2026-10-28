# Fig. 2--4 single-column trial audit

## Scope

Fig. 1 remains a two-column `figure*`. Figs. 2--4 use ordinary `figure` floats at full `\columnwidth`. Frozen PDFs, captions, scientific text, claims, and gates are unchanged.

## Compilation and placement

- Trial pages: **4**.
- REVTeX column width: **246.00 pt**; text width: **510.00 pt**.
- First-text pages: `{'1': 1, '2': 2, '3': 3, '4': 3}`.
- Compiled figure pages: `{'1': 2, '2': 3, '3': 3, '4': 4}`.
- Placement: **PASS** under the within-two-pages structural rule.
- Overfull boxes: **0**; underfull boxes: **0**.

## Print-scale projection

| Figure | Native width | Placed width | Scale | Projected 7 pt internal text | Status |
|---|---:|---:|---:|---:|---|
| FIG-02-THERMO | 504.57 | 246.00 | 0.487 | 3.41 pt | HUMAN REVIEW |
| FIG-03-STAGEIII-TOPOLOGY | 504.57 | 246.00 | 0.487 | 3.41 pt | HUMAN REVIEW |
| FIG-04-STAGEIV-3D-EXTENSION | 504.57 | 246.00 | 0.487 | 3.41 pt | HUMAN REVIEW |

The projection is a geometry check, not an automatic aesthetic decision. Internal labels near 3.4 pt are below the Figure Engine's registered 7 pt native/full-width target, so the single-column trial cannot receive an automatic readability pass. The final decision requires inspection of the rendered pages at native letter size and a 100% print-scale crop.

## Rendered visual review

Status: **PASS AS A LAYOUT PROOF; FAIL AS A FINAL DIRECT-SCALE ARTWORK**.

- Fig. 2: the thermodynamic phase structure and allocation pattern remain visible, but convergence-panel labels and small numerical annotations are below comfortable print readability.
- Fig. 3: the cFFLO/tFFLO map remains visually legible, but the right-hand convergence audit and k-sensitivity values are too small at native single-column print scale.
- Fig. 4: the 3D point-cloud structure remains visible, but fixed-$\mu$ slice labels and strict-gate numerical values are too small at native single-column print scale.
- No clipping, overlap, broken glyph, or float-placement defect is visible in the four rendered pages.

The four-page reading flow is substantially better than the six-page all-double-column candidate. The recommended implementation is therefore to keep the single-column manuscript layout but regenerate Fig. 2--4 as **single-column-native 86.8 mm assets** with at least 7 pt internal text, simplified secondary labels, and unchanged data/caveats. Directly scaling the current 178 mm freezes should remain a trial only.

## Provenance

- Source candidate SHA-256: `5846a2b4ec76a25e4824a012aa45f8451a92ccf1d743e5c97b61576b452af222`.
- Trial candidate SHA-256: `7cbcd6ba4d8fd629d0116c45edb0ede44f765f79f402c704b5273c32ac9877c1`.
- All four source and staged figure hashes match the canonical frozen registry.
- No external retrieval or physics calculation was performed.
- This trial does not change or pass `GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW`.
