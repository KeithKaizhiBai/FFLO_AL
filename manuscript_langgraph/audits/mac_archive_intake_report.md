# Mac scientific archive intake report

Date: 2026-08-13

## Intake result

`active_learning_scientific_evidence_archive.zip` is accepted as a read-only
scientific evidence source for manuscript figure regeneration. The archive
SHA-256 is
`8f3402bb7dce090eac3504adff3cf3cfd55ab24564a309c6042d94bf3311027d`.
It contains 2,397 ZIP members; no duplicate member, unsafe path, or symbolic
link was found. Decompression passed. All 2,396 files listed in the internal
checksum file verified after normalizing its CRLF line endings.

The source manifest records Windows source commit
`8405b03639043bbc5dc43e154ed7f7465e77cc75`. The archive was extracted under
`downloads/active_learning_scientific_evidence_archive_20260813/` and is not
modified by manuscript scripts.

## Canonical artifact selection

| Stage | Canonical artifact | Resolution |
|---|---|---|
| I history | `active_boundary_discovery_rankcap_k3_full_loop_v1/dataset_iter031.npz` and selection history | used only for cold-start/acquisition-history panels |
| II | `active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/dataset_iter035.npz` | authoritative thermodynamic endpoint |
| III | `active_phase_topology_from_scratch_full_loop_v2/dataset_iter018.npz` | authoritative topology-aware cold-start endpoint |
| III validation | `topology_pass_dataset_iter035_v1` | separate oracle/Pfaffian validation role; not a competing endpoint |
| IV | `active_phase_topology_3d_t_ja_mu_from_scratch_v1/dataset_iter025.npz` | authoritative 3D endpoint |
| IV figure repair | `stageiv_curve_extraction_fix_local` | derived from the same endpoint; not a competing run |

No conflicting authoritative candidate was found, so
`GATE_CANONICAL_ARTIFACT_SELECTION` is not triggered.

## Scientific integrity checks

- Stage II and Stage IV endpoint hashes match their reproduction manifests.
- Stage II, III, and IV endpoint NPZ arrays match their paired CSV arrays.
- Stage III convergence tables, figure sources, and decision record are local.
- The Stage III frozen decision JSON labels its source as NPZ but stores the
  paired CSV hash. The per-iteration provenance table names the CSV correctly;
  the frozen JSON remains unchanged.
- The Stage IV curve-extraction repair removed unsupported broad-bin smooth
  segments without changing labels or convergence status.

## Use policy

Manuscript scripts read the stable paths in `artifact_path_map.yaml`. Large
archive data are not copied again. Existing lightweight endpoint recovery
copies remain the canonical Mac-side manuscript inputs and retain their source
hashes. No physics recomputation, exact call, HPC submission, or frozen-label
edit is authorized by this intake.
