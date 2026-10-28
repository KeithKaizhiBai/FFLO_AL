# Manuscript provenance audit

The canonical physical definitions are `MODEL_SPEC.md` and
`docs/NUMERICS_SPEC.md`. Decision status is taken from `docs/DECISIONS.md` and
the latest dated Q&A, with later records overriding earlier planning notes.

## Accepted provenance layers

1. Frozen raw dataset, run manifest, and exact output.
2. Reproducible convergence table/figure derived from a frozen run.
3. Signed-off audit/decision record with stable run identity.
4. Consolidated report or Q&A when the underlying artifact is unavailable.
5. Planning prompt, which may constrain wording but is never scientific
   evidence.

## Recovered provenance

The Stage II--IV claims now reach layers 1--3 through the verified evidence
archive. Endpoint NPZ/CSV pairs agree field-for-field, report counts agree with
the arrays, and Stage II/IV endpoint hashes match their reproduction manifests.
The Stage III convergence bundle retains its final endpoint, decision, tables,
boundary data, plotting source, and original generated figures.

The Stage III decision JSON contains a metadata-label mismatch: it names the
NPZ but stores the correct CSV hash. The per-iteration provenance table names
the CSV correctly, and NPZ/CSV values agree. This is documented rather than
silently corrected.

## Isolation

No QAHE3D file is modified or imported at runtime. The adapted Introduction
contract records upstream paths and SHA-256 hashes in
`vendor/qahe3d_intro_citation_adapter/UPSTREAM_PROVENANCE.md`. There is no
two-way synchronization.

## Forbidden provenance operations

- merging Stage III and Stage IV/V datasets;
- treating a smoothed 3D diagnostic surface as a frozen exact boundary;
- promoting Stage V/VI development metrics to central physics evidence;
- using PRL style prompts as scientific evidence;
- editing recovered frozen evidence in place;
- hiding the Stage III hash-label note or Stage IV non-convergence.

## Current gate consequence

The artifact/provenance and evidence-level publication-readiness gates pass.
`GATE_SCOPE_APPROVAL` and `GATE_CLAIM_APPROVAL` are approved. The active workflow
still stops at `GATE_FIGURE_APPROVAL`: recovered data authorize reproducible
redraw and review, not automatic figure approval. External Introduction
retrieval remains unauthorized.
