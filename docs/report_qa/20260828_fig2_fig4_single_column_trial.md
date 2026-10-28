# Fig. 2--4 single-column trial

## Question

Can the approved Fig. 2--4 artwork be inserted as single-column figures in the
PRL-style integrated manuscript?

## Answer

The manuscript layout can use that structure, but the present 178 mm artwork
should not simply be scaled down. Keeping Fig. 1 double-column and placing
Figs. 2--4 in single columns reduces the candidate from six pages to four and
gives a substantially better reading flow. Fig. 2 and Fig. 3 can share one
page, and Fig. 4 can sit beside Discussion without float or compilation
failures.

At the actual REVTeX column width of 246 pt, however, each 504.57 pt figure is
scaled by 0.4875. The Figure Engine's registered 7 pt internal labels become
approximately 3.41 pt. The main phase maps and point clouds remain visually
recognizable, but several mandatory numerical details are not comfortably
readable: the small convergence panels in Fig. 2, the k-sensitivity and audit
panels in Fig. 3, and the fixed-mu/nonclosure annotations in Fig. 4.

Therefore the scientific layout recommendation is:

```text
Fig.1: retain double-column artwork
Fig.2--4: use single-column placement
Fig.2--4 assets: regenerate natively at 86.8 mm
minimum internal text: 7 pt at placed size
directly scaled current freezes: trial only
```

The native single-column rebuild may simplify repeated labels or transfer
supporting detail to the unchanged caption/Supplement, but it must retain the
Stage III k=6 sensitivity and the Stage IV finite-domain, slice, and strict
nonclosure evidence.
