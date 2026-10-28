# Single-column placement Q&A

## What changed?

Fig. 1 remains a double-column figure. Figures 2--4 are now ordinary
single-column REVTeX floats. Each result block ends with a legal float barrier
so the next result does not advance before the corresponding figure is placed.

## Did the scientific content change?

No. The physical prose, equations, captions, bibliography, Supplement,
frozen figure binaries, datasets, labels, and claim scope are unchanged.

## Where do the figures compile?

- Fig. 1: page 3, one page after its model discussion.
- Fig. 2: page 4, on the same page as its discussion.
- Fig. 3: page 4, on the same page as its discussion.
- Fig. 4: page 5, one page after its first discussion and immediately beside
  the end of its result block/Discussion transition.

## What remains uncertain?

The existing Figures 2--4 were designed at 178 mm and are now directly scaled
to one-column width. Their fine internal labels are visibly smaller. The
layout satisfies the requested column mode and proximity, but print-size
readability still requires human acceptance or later native 86.8-mm figure
derivatives.

## Gate

`GATE_SINGLE_COLUMN_FIGURE_PLACEMENT_HUMAN_REVIEW = HUMAN_WAIT`.
