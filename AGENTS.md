## Project nature

This is a computational physics research project. Correctness is more important than speed or elegance.

## Required reading before coding

Before changing code, read:

1. docs/PROJECT_CONTEXT.md
2. docs/MODEL_SPEC.md
3. docs/NUMERICS_SPEC.md
4. docs/DECISIONS.md
5. The target source files related to the task

Do not modify code before summarizing your understanding of the relevant model, assumptions, and expected behavior.

## Work rule

For non-trivial tasks:

1. First inspect the codebase.
2. Identify the exact files involved.
3. Produce a short implementation plan.
4. State what will not be changed.
5. Implement in small diffs.
6. Run relevant tests.
7. Review the diff.
8. Update docs if behavior or assumptions changed.

## Scientific constraints

- Do not silently change physical definitions.
- Do not change parameter conventions without updating docs/MODEL_SPEC.md.
- Do not introduce heuristic post-processing unless explicitly requested.
- Do not replace a physics-based criterion with a visual or empirical shortcut.
- Preserve reproducibility: every generated figure must record parameters and commit hash when possible.

## Completion criteria

A task is done only when:

1. The code runs.
2. Relevant tests pass.
3. Numerical results are checked against at least one known limit, benchmark, or previous result.
4. The final response explains what changed, what was verified, and what remains uncertain.

## Git/GitHub rules

- Always run `git status` before commit or push.
- Never commit `.env`, API keys, credentials, model keys, or private tokens.
- Do not commit large generated outputs under `output/` unless explicitly requested.
- Do not push unless I explicitly ask.
- Before creating a repository, ask whether it should be public or private if not specified.
- Prefer private repositories by default.
- Summarize changed files before committing.


## Project-specific memory

This project must maintain:

- `docs/PROJECT_SUMMARY.md`
- `docs/MODEL_SPEC.md`
- `docs/NUMERICS_SPEC.md`
- `docs/DECISIONS.md`
- `docs/report_qa/` for question-and-answer notes intended for later report
  integration

Before modifying code, read these files if they exist.

After any meaningful change, update `docs/PROJECT_SUMMARY.md`.

When a discussion produces explanatory text that may be useful for the final
report, paper draft, thesis text, or presentation narrative, save or summarize
it under `docs/report_qa/`. Use lowercase, cross-platform filenames such as
`YYYYMMDD_topic.md`. These Q&A notes are supporting report material and must
not silently override the canonical physical definitions in `MODEL_SPEC.md`,
the numerical rules in `docs/NUMERICS_SPEC.md`, or the decisions in
`docs/DECISIONS.md`.

For research-code changes, also update:

- `docs/MODEL_SPEC.md` if physical definitions changed
- `docs/NUMERICS_SPEC.md` if numerical methods changed
- `docs/DECISIONS.md` if an important design decision was made

## Formula notation for explanatory material

For project webpages, reports, slides, workflow diagrams, and other explanatory
materials, write mathematical formulas in LaTeX notation. Avoid ASCII
pseudo-formulas for physical or ML equations unless the text is explicitly
showing code, filenames, command-line arguments, or raw configuration keys.

## Report synchronization and ChatGPT handoff protocol

This project often uses Codex to run numerical audits, generate reports, and
then ask ChatGPT to interpret the results. To make this reliable, every
report-generation task must produce machine-readable companion files in
addition to any PDF.

### 1. Never rely on PDF alone

When generating a report, always output:

```text
<report_name>.pdf          # human-readable final report
<report_name>.md           # full Markdown report with the same scientific content
tables/*.csv               # all key numerical tables
figures/*.png              # all important figures
decision_log.md            # short decision-level summary
```

The Markdown report must preserve the scientific definitions, assumptions,
main numerical results, and caveats from the PDF. The decision log should be
short and should state what was concluded, what remains unresolved, and what
the next calculation should check.
