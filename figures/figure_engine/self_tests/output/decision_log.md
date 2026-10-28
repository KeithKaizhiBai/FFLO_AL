# Figure engine self-check decision log

## Conclusion

`FIGURE_ENGINE_HEALTH = pass_with_known_fig1_failure`.

The engine reads and hashes its rules, constrains engine-native generation,
detects all 15 intentional negative-control violations, admits the corrected
control, and preserves a stable Figure 1 regression signature.

## Unresolved

- Figure 1 v5 has one 3.25-px text-border clearance against a 4-px minimum.
- The legacy Figure 1 builder still contains local tokens despite mandatory
  preflight; future engine regeneration needs a full token adapter or an
  engine-native builder.
- Perceptual hierarchy and physical truth remain human/figure-specific review
  responsibilities.

## Next authorized step

Do not redraw Figure 1 yet. After explicit authorization, create a non-final
`fig1_engine_test_v1` benchmark that consumes engine-native tokens and must pass
the generic lint, print-size checks, the Fig.1-specific scientific audit, and a
human comparison against v5.
