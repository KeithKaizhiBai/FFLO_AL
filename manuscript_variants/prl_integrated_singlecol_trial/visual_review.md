# Single-column trial visual review

## Review basis

The four compiled letter-size pages were rendered at 200 dpi and inspected in
full. Fig. 2, Fig. 3, and Fig. 4 were also rendered as separate 300 dpi
single-column crops. This review concerns composition and print readability;
it does not alter the frozen scientific evidence.

## Findings

### Manuscript flow

The single-column placement is structurally successful. The manuscript falls
from six pages to four, Fig. 2 and Fig. 3 occupy the two columns of page 3, and
Fig. 4 shares page 4 with Discussion. There is no figure-at-end behavior,
clipping, overlap, broken glyph, or disruptive blank figure page.

### Fig. 2

The large thermodynamic endpoint and the cold-start/allocation panels remain
recognizable. Axis labels, the four small convergence panels, and numerical
annotations are substantially reduced. They can be read when zoomed but not
comfortably at native print scale.

### Fig. 3

The cFFLO/tFFLO point map remains the dominant and readable object. The
gap-closing diagnostic, four convergence plots, and k-sensitivity values are
too small at native print scale. This matters because the $k=6$ sensitivity is
a mandatory scientific caveat, not decorative detail.

### Fig. 4

The 3D thermodynamic point cloud remains interpretable. The topology inset,
four fixed-$\mu$ slice labels, sample counts, and strict convergence metrics
are too small at native print scale. These elements carry the nonclosure
caveat and must remain directly legible.

## Decision

```text
LAYOUT_PROOF = PASS
DIRECT_SCALE_AS_FINAL = FAIL
RECOMMENDED_LAYOUT = Fig.1 double column; Fig.2--4 single column
REQUIRED_ASSET_ACTION = regenerate Fig.2--4 at native 86.8 mm contracts
MINIMUM_INTERNAL_TEXT = 7 pt at placed size
SCIENTIFIC_DATA_OR_CAVEAT_CHANGE = forbidden
HUMAN_GATE = pending
```

The single-column idea should be retained, but the present 178 mm frozen PDFs
should not be adopted merely by scaling. A separately gated Figure Engine pass
should produce single-column-native variants that simplify secondary labels
without removing data, sensitivity tests, or the Stage IV nonclosure evidence.
