# Report Audit Template

Use this template for report-only analysis from existing results.

## Scope

State which result directory, dataset, and iteration are frozen inputs.

## Required Reading

List reports, decision logs, manifests, configs, and relevant documentation.

## Analysis Questions

1. What are the final counts and status metrics?
2. Which figures are needed?
3. Which convergence or reliability metrics are required?
4. What caveats must be preserved?

## Required Outputs

```text
report.md
report.pdf
decision_log.md
tables/*.csv
figures/*.png
manifest.json
```

## Validation

Check that Markdown, PDF, tables, and figures agree numerically. Render the PDF
when possible and inspect representative pages.
