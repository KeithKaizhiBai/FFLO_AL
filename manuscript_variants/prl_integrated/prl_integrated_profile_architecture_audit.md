# PRL-integrated profile architecture audit

## Finding

The existing manuscript graph already supplied shared state, provenance/readiness checks, claim and figure gates, frozen Results/captions, canonical Figure Engine sources, and REVTeX compilation. It did not contain publication profiles, method packets, inline placement, semantic-equivalence tracing, or a length-budget audit.

## Minimal extension

The new graph reuses `artifact_provenance_audit`, `publication_readiness_gate`, `claim_freeze`, and `figure_evidence_contract`. It adds profile selection, packetization, weaving, inline placement, assembly, semantic audits, length audit, and one human gate. The existing 17-node graph and its nodes remain unchanged.

Both variants load the same claim/evidence/figure registries and canonical Figure Engine source map. The new run has parent provenance and writes only to `active_learning_prl_integrated_v1`, `manuscript_variants/prl_integrated`, and `output/pdf/manuscript_prl_integrated_v1`; it does not overwrite the parent run or intermediate PDF.
