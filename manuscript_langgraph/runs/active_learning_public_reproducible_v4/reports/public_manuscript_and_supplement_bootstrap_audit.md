# Public manuscript and Supplement bootstrap audit

## Current architecture

The repository contains an evidence-first manuscript graph with claim, evidence,
and figure registries; versioned manuscript variants; persistent run records; and
human approval gates.  The current parent is `active_learning_manuscript_stageiv_revision_v3`, whose main and
Supplement candidates are preserved unchanged.

## Diagnosis

The parent main text inherited sentence-level audit language and full count or
threshold dumps because evidence presence was previously treated too directly as
publication visibility.  Its Supplement was assembled chiefly from validated
reports and therefore did not ingest the implementation-level model, training,
acquisition, restart, and exact-call-accounting evidence needed for independent
reproduction.

## Reusable nodes

- registries and frozen figure authority;
- citation and LaTeX integration;
- sentence-level observation/interpretation trace;
- human review gates and versioned run directories.

## Minimal additions

- publication visibility routing;
- public-language adaptation and main-text information-budget checks;
- frozen-artifact efficiency accounting;
- implementation-first method evidence ingestion;
- a Supplement reproducibility completeness gate.

## Source authority

Implementation evidence is read in the order source code, configuration, run
manifest, validated report, and inherited prose.  Conflicts are recorded rather
than merged silently.

## Frozen boundaries

Figures 1--4, their data, the physical Hamiltonian, phase conventions, numerical
tolerances, BdG outputs, and production active-learning runs are not modified.
The export is not a Git worktree, so a commit identifier is unavailable; frozen
asset SHA-256 values are used instead.

## Baseline tests

The manuscript LangGraph baseline passed 40/40 tests.  The broader project suite
passed 113 tests and had seven pre-existing failures caused by absent local-
refinement packaging artifacts; none concerns this manuscript workflow.
