# Figure-engine self-tests

These fixtures test the framework; they are not manuscript figures.

- `invalid_figure/` intentionally violates typography, geometry, hierarchy,
  color, arrow, annotation, scientific-text, primary-subobject, and
  scientific-content fill-ratio rules.
- `corrected_figure/` is the matched passing control.
- `fig1_regression_expectation.yaml` freezes the current read-only Figure 1
  failure signature.
- `run_validator_self_test.py` asserts negative/control behavior.
- `run_fig1_scientific_asset_fill_audit.py` detects the known frozen-v1
  surface-fill failure without modifying artwork.
- `run_figure_engine_health_check.py` runs the complete non-rendering health
  check and writes the canonical health report at the engine root.

Never remove an expected failure merely to obtain green output. A rule change,
fixture repair, artwork repair, and regression-expectation update are distinct
decisions and require separate justification.
