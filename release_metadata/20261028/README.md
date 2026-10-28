# Release inventory and publication receipt

The source/data/manuscript payload was published as commit
`b0c97936e6a8a989ebed194e9c4c18a699f22ba3` on `main`.

- `file_manifest.csv` records selected source-file byte hashes. The version at
  the payload commit describes exactly that selection; the current version
  also covers the documentation-only publication receipt.
- `summary.json` records the local packaging operation, including exclusions.
  Its `local_copy_complete_not_committed_or_uploaded` status is deliberately
  the packager's status at generation time, not the live GitHub publication
  state. The packager never commits or uploads.
- [Publication receipt](../../docs/github_release_20261028.json) records the
  successful payload push and seven representative remote SHA256 checks.

The inventory excludes virtual environments, credentials, cluster-login
documents, caches and duplicate large archives. It does not imply complete
Stage VI replay, cross-machine Figure Engine verification, or scientific
approval. See the repository README and current author-review document.
