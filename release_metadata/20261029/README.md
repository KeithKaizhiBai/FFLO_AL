# Clean manuscript synchronization — 2026-10-29

This update publishes the reader-facing manuscript collection under
`reports/manuscript/v1` through `v9`, its central PDF index, the mechanical
export checker, regression tests and audit records. It does not change
scientific data, the original PDFs or frozen figure artwork, and deletes no
existing repository files.

`file_manifest.csv` records the allowlisted source selection and byte hashes.
`summary.json` describes the local packaging operation; its pre-publication
status is not a live GitHub status. See `publication.json` for synchronization
status and remote verification. The dated 20261028 inventory is preserved.

The nine version folders contain only required TeX, original PDFs, referenced
figures and bibliography files. The local `reports/latest` shortcut is not
uploaded; the portable Markdown index points directly to the v9 directory.
113 manuscript/export tests passed and all 18 documents previously passed
clean compilation, page-text and 72-dpi full-page RGB comparison.

User authorization is limited to repository synchronization; no remaining
scientific or figure-approval gate is advanced.
