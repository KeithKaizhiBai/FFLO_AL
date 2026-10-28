# Figures 1--4 contract and Stage III vertical-slice review packet

Date: 2026-08-12  
Approved scope: `AL_FFLO_PRL_RELIABILITY_AWARE_3D_V2`  
Claim gate: `GATE_CLAIM_APPROVAL = approved`  
Current workflow stop: `GATE_FIGURE_APPROVAL`

This packet does not infer figure approval. It presents the approved claim
positions, Stage III Results vertical slice, and revised main-figure contracts
for the next human decision. The frozen sources for Figures 2--4 were recovered
and verified on 2026-08-13; manuscript redraw and approval remain pending.

## A. Approved claim positions

| Claim | Status | Destination | Approved boundary |
|---|---|---|---|
| `CLM-2D-THERMO` | `frozen_supported` | Main text | no change |
| `CLM-ORACLE-RELIABILITY` | `supported_with_caveat` | Main + Supplement | enabling scalable active learning and 3D, without mathematical-exactness language |
| `CLM-2D-TOPOLOGY` | `supported_with_caveat` | Main text | $k=8$ pass, $k=6$ sensitivity, diagnostic-only brackets |
| `CLM-3D-EXTENSION` | `supported_with_caveat` | Main text / candidate Fig. 4 | higher-dimensional exploration; strict 3D topology closure not passed |
| `CLM-LEARNED-ACQUISITION` | `discussion_only` | Discussion | exploratory; no superiority |
| `CLM-GROUNDUP-WORLD-MODEL` | `future_work` | Outlook | future learned physical world models; no validated EIG |

Canonical registry: `manuscript_langgraph/registries/claim_registry.yaml`.

## B. Main-figure contracts

| Figure | Scientific role | Current reproducibility state |
|---|---|---|
| Fig. 1 | reliability-aware active-learning framework | ready for schematic redraw after figure approval |
| Fig. 2 | 2D thermodynamic reconstruction | evidence recovered; pending contract approval and redraw |
| Fig. 3 | topology-aware cFFLO/tFFLO discovery | evidence recovered; pending approval/redraw; retain CSV-hash metadata note |
| Fig. 4 | higher-dimensional 3D extension | evidence recovered; pending approval/redraw; strict-convergence caveat mandatory |

Figure 4 must use “higher-dimensional exploration” / “3D extension” and may
state that it reveals thermodynamic and topology-aware structures. It must not
say “complete 3D phase diagram” or “fully converged 3D topology atlas.”

Canonical registry: `manuscript_langgraph/registries/figure_registry.yaml`.

## C. Stage III vertical slice

Review in this order:

1. `vertical_slice_stageIII/claim_contract.yaml`;
2. `vertical_slice_stageIII/evidence_extract.md`;
3. `vertical_slice_stageIII/figure_contract.yaml`;
4. `vertical_slice_stageIII/caption_candidate.md`;
5. `vertical_slice_stageIII/results_candidate.md`;
6. `vertical_slice_stageIII/paragraph_metadata.json`;
7. `vertical_slice_stageIII/audit_report.md`.

Facts encoded in the slice include 4,345 final samples, 3,213 trusted gapped
FFLO points (2,290 cFFLO and 923 tFFLO), configured $k=8$ Decision A,
$p_{95}=0.006185$ against 0.00625, and the $k=6$ sensitivity value 0.006269.
Opposite-$Z_2$ brackets remain support diagnostics rather than topology
contours.

## D. Requested gate decision

At `GATE_FIGURE_APPROVAL`, either:

- approve the Stage III slice and Figures 1--4 contracts as written;
- request named changes; or
- leave the gate pending.

Approval of contracts authorizes the evidence-locked redraw workflow; it does
not automatically approve the resulting scientific panels. External
Introduction retrieval remains forbidden until separately granted after
Results/figure claims are frozen.
