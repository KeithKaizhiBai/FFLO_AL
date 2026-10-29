# Minimal manuscript versions and reproducibility checks

Date: 2026-10-29

## User-facing result

The paper-reading entry is now `reports/README.md`. Clean v1–v9 packages live
under `reports/manuscript/`; `reports/latest` points locally to v9. Each version
contains only required TeX, compiled PDFs, actually referenced figure assets
and bibliography files. Main/SM/review are documents within a version, not
separate versions. v1 means the first cited integrated manuscript; v6's SM
remains inherited from v5 and is linked in the central index.

The nine selected versions contain 18 PDF documents and 158 files, totaling
68,557,332 bytes. The corresponding mixed workflow directories contained 504
files and 153,473,273 bytes. These are package sizes, not disk space reclaimed:
the original provenance/archive directories remain intact.

## What was changed and preserved

Added the mechanical export/verification script and focused regression tests.
Entry TeX names and local asset paths were normalized only in exported copies.
Required helper TeX files are grouped under `tex/`. Source, data, canonical
artwork, original document PDFs, bibliography bytes and all prior Gate states
are unchanged. Every transitive TeX input is included in the export audit.
The export refuses changed or symlinked destination versions rather than
overwriting user edits. The old v7/v8 every-file hash guards remain valid.

The cleanup phase performed no new physics, literature retrieval,
scientific-figure regeneration, scientific rewriting, Git commit or upload.
The user subsequently authorized GitHub synchronization as a separate step;
its inventory and publication receipt are under `release_metadata/20261029/`.
Updating a packaging allowlist alone is not proof of remote publication.

## Verification

- All nine versions compile from their own self-contained dependencies in
  clean temporary directories; no old `.aux` or `.bbl` is needed.
- All 18 PDFs have identical per-page extracted text and identical full-page
  RGB pixels at 72 dpi to the corresponding original saved PDFs.
- Exported PDFs, figures and Bib files are original byte-for-byte copies.
- No unresolved references or citations remain in the final clean builds.
- 113 manuscript/export tests pass, including parent guards, dependency
  closure, used-asset inventory, cross-document references, transitive input
  hashes, no-overwrite behavior and nested bibliography detection.
- The updated GitHub packager passed a read-only selection/secret-pattern
  check. `reports/latest` is only a local shortcut; portable uploads use the
  actual version directories and Markdown links, not that symlink.

Detailed machine-readable records: `20261029_manuscript_exports/v1.json`
through `v9.json`. Original Markdown/CSV/decision-log companions remain in
workflow storage; they were not discarded to satisfy the minimal reader view.

## Two implementation lessons

`latexpand` changed whitespace at some input boundaries, causing line-wrap
drift despite unchanged scientific wording. That preliminary export was
rejected. Preserving the necessary TeX subfiles restored equivalence.

The historical compiler uses TeX Live's restricted command mode. Explicitly
disabling it with `-no-shell-escape` changed spacing on one line of v9 SM page
13. Reversible A/B checks on the same source and auxiliary files isolated the
flag; restoring restricted mode restored exact content streams and pixels.
The final checker uses `-shell-restricted`, never unrestricted
`-shell-escape`. No external `runsystem` invocation appeared in the inspected
logs. The lower-level pdfTeX/microtype cause was not established; this is a
bounded environment observation, not a physical change or a relaxed visual
criterion.

Future revisions should be authored in the existing Manuscript LangGraph and
Figure Engine, then exported as a new reader version. Do not silently edit an
old version, replace saved PDFs with temporary builds, or treat packaging and
software tests as scientific approval.
