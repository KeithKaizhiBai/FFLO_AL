# Phase 1 plan - repository and archive audit

## Audit

Inspect the Mac manuscript package, archive manifest/checksums, Stage I--IV
lineage, canonical endpoint candidates, and existing LangGraph entrypoints.

## Plan

1. Verify archive identity and read-only location.
2. Map canonical endpoints by run identity and manifest provenance, not mtime.
3. Distinguish validation/derived artifacts from competing endpoints.
4. Record stable artifact IDs and hashes.

## Critique

The primary risks are choosing a same-numbered dataset from the wrong run,
treating the Stage III topology-pass dataset as the cold-start convergence
endpoint, and treating the Stage IV curve repair as a new scientific run.

## Non-changes

No frozen source, BdG code, physical criterion, exact calculation, HPC state,
QAHE3D file, or external literature source will be changed or invoked.
