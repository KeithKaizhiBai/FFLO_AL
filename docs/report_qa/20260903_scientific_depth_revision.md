# Scientific-depth manuscript revision Q&A

## What was missing from the public manuscript?

The parent manuscript had a complete reproducibility layer but compressed the
Results into caption-like summaries.  The revision adds the compact
altermagnetic single-particle model, the joint $\operatorname*{arg\,min}_{\Delta,q}$
logic, the thermodynamic/topological hierarchy, and parameter-dependent
interpretation for Figures 2--4.

## What does the production free-energy expression mean?

The code sums all four BdG eigenvalue branches weighted by the Fermi
occupation, applies the $1/2$ Nambu factor, adds $\Delta^2/U$, and subtracts the
$\Delta=0$ spectral expression at the same $q$, normalized by the full
$N_k=800$ grid.  There is no separately coded entropy or logarithmic partition
term.  The manuscript therefore calls the quantity $F_{\rm impl}$, the
implemented normal-referenced occupation-weighted spectral objective.  This is
a notation clarification, not a recomputation or change of minima.

## Why can 6,880 and 7,434 both be correct?

They refer to different endpoints.  Stage I retains 6,880 unique exact
coordinates from the prescribed 77,361-location grid and supports the 8.893%
and 11.244-fold location-economy headline.  Stage II retains every Stage-I
coordinate and 554 additional exact coordinates, giving the 7,434-coordinate
endpoint displayed in Fig. 2(c).  Shared coordinates have identical phase
labels.  Trusted subsets are 6,104 and 6,598, respectively.

## What physical structure is now stated explicitly?

- Fig. 2: a descending curved Normal--superconducting envelope, an extended
  finite-$q$ sector, and a narrow uniform-SC strip near $J_A=0$.
- Fig. 3: cFFLO and tFFLO are distinct gapped bulk sectors inside the same
  thermodynamic FFLO sector; the index changes only through a gap
  close-and-reopen sequence, while the plotted contour remains a guide.
- Fig. 4: the sampled support reorganizes across the four fixed-$\mu$ cuts;
  the last cut contains no trusted sampled tFFLO point, but this is not a
  global-absence claim or a continuous-surface reconstruction.

## Why is the figure gallery still present?

The four canonical assets were designed and validated at 178 mm.  Legal
full-width REVTeX floats preserve their readability but queue on four
consecutive pages.  A legal mixed single-column trial reduces the gallery but
scales Figures 2--4 to about 47%, making their internal text unreadable.
Resolving the tradeoff requires native single-column figure derivatives or a
different layout, both of which require separate human authorization.

## Current decision state

`GATE_EXPANDED_PUBLIC_MANUSCRIPT_HUMAN_REVIEW = HUMAN_WAIT`.  No scientific or
publication freeze has been inferred.
