# Claim registry and Stage III vertical-slice review packet

Date: 2026-08-08  
Approved scope: `AL_FFLO_PRL_PHYSICS_FIRST_V1`  
Current workflow stop: `GATE_CLAIM_APPROVAL`

This packet presents the claims and the already prepared Stage III vertical
slice for scientific review. It does not record approval. Figure 1–3 expansion,
external Introduction retrieval, and Figure 4 selection remain inactive.

## A. Claim-registry review

Canonical registry: `manuscript_langgraph/registries/claim_registry.yaml`

| Claim | Proposed status | Destination | Scientific review focus |
|---|---|---|---|
| `CLM-2D-THERMO` | `frozen_supported` | Main text | Is report-level evidence sufficient to freeze wording while final Figure 2 remains blocked by the missing endpoint? |
| `CLM-ORACLE-RELIABILITY` | `supported_with_caveat` | Main text + Supplement | Does “numerically audited exact-oracle hierarchy” correctly preserve finite-resolution and unresolved-state caveats? |
| `CLM-2D-TOPOLOGY` | `supported_with_caveat` | Main text | Is Decision A under the configured audit stated conservatively enough, including the tight k=6 coverage caveat? |
| `CLM-3D-EXTENSION` | `provisional` | Conditional Supplement/Discussion | Confirm that Stage IV remains explicitly non-converged and is not yet authorized as Figure 4. |
| `CLM-LEARNED-ACQUISITION` | `discussion_only` | Discussion | Confirm that Stage V is an operational, negative comparative result with no V4-superiority claim. |
| `CLM-GROUNDUP-WORLD-MODEL` | `future_work` | Outlook | Confirm that EIG remains unauthorized and the spectral-descriptor gap is explicit. |

Requested gate decision: approve the registry as written, request named edits,
or leave `GATE_CLAIM_APPROVAL` pending. No approval is inferred from reviewing
this packet.

## B. Stage III vertical-slice review

Review in this order:

1. `vertical_slice_stageIII/claim_contract.yaml` — allowed numbers, caveats,
   and forbidden wording;
2. `vertical_slice_stageIII/evidence_extract.md` — evidence hierarchy and
   provenance limitation;
3. `vertical_slice_stageIII/figure_contract.yaml` — four proposed panels and
   current blocked/report-only status;
4. `vertical_slice_stageIII/caption_candidate.md` — caption claim boundary;
5. `vertical_slice_stageIII/results_candidate.md` — three Results paragraphs;
6. `vertical_slice_stageIII/paragraph_metadata.json` — sentence-level evidence
   trace;
7. `vertical_slice_stageIII/audit_report.md` — retained blockers and gate.

Scientific facts currently encoded in the slice:

- final cold-start sample count: 4345;
- trusted gapped FFLO points: 3213, split into 2290 cFFLO and 923 tFFLO;
- configured k=8 audit: Decision A;
- final coverage p95: 0.006185 against 0.00625;
- k=6 sensitivity: 0.006269, slightly above the nominal threshold;
- local opposite-Z2 brackets are support diagnostics, not the topology contour;
- the raw endpoint and convergence/figure bundle remain absent after the
  2026-08-08 archive/export search.

Requested later gate decision: after the claim registry is approved, approve
the Stage III slice as written, request named edits, or leave
`GATE_FIGURE_APPROVAL` pending.

## C. Locked downstream behavior

- Figures 1–3 are not expanded until both claim and figure gates are approved.
- Figure 4 remains conditional: matched multi-seed benchmark preferred;
  otherwise an explicitly non-converged Stage IV demonstration; otherwise omit.
- External Introduction retrieval remains forbidden until Results claims are
  frozen and permission is explicitly granted.
