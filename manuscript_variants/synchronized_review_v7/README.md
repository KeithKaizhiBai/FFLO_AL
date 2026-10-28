# Synchronized author-review candidate

Parent prose: scientific_depth_v5; parent figure placement: scientific_depth_single_column_v6.
This profile supersedes v6's figure-size instruction only for this candidate, at the author's explicit 2026-09-11 request.

Build from the project root:

```sh
world_model_descriptor_dev/.venv/bin/python manuscript_langgraph/scripts/run_single_column_placement.py --profile synchronized_v7
```

No physics or Figure Engine artwork is regenerated. The existing four-node graph is reused with a paired-build profile. Main figures bind to the versioned authority in `review_agent/figure_authority.py`, never filename ordering. The portable output contains a standalone clean build script; no previous auxiliary files are required.

Human review remains mandatory, including the unresolved mechanism wording listed in AUTHOR_REVIEW.md. No authors, affiliations, funding, public repository, or DOI have been invented.
