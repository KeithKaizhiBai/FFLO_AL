# Stage III vertical-slice audit

## Outcome

`PASS_TO_HUMAN_GATE`

The Results-by-figure path is complete from frozen claim contract through
evidence extraction, caption, Results prose, paragraph metadata, consistency
stub, and isolated LaTeX fragment. No unsupported sentence is recorded.

## What passed

- every numerical statement maps to `E-STAGEIII-PASS` or
  `E-STAGEIII-CONVERGENCE`;
- observation and interpretation are separated;
- the configured convergence decision and sensitivity caveat coexist;
- forbidden wording is absent;
- the LaTeX fragment contains an explicit non-production warning.

## Remaining gate items

- Stage III final dataset, convergence tables, original figures, and source are
  recovered and verified;
- manuscript panels remain contracts/placeholders until a versioned redraw is
  generated from the recovered evidence;
- claim-registry approval is recorded; vertical-slice/figure approval remains
  a human decision;
- the frozen decision JSON's NPZ-labeled hash is the paired CSV hash; the
  per-iteration table is correct and NPZ/CSV arrays agree.

Gate: `GATE_FIGURE_APPROVAL` (Stage III vertical slice plus revised Figures
1--4 contracts).
