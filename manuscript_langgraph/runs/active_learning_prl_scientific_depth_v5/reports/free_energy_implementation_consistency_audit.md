# Free-energy implementation consistency audit

## Status

`PASS_WITH_NOTATION_FIX`

## Code-to-equation trace

- `bdg_hamiltonian_batch` returns all four Hermitian BdG eigenvalue branches.
- `fermi_weighted_sum` evaluates `sum_n E_n sigmoid(-E_n/kT)` over those four branches.
- `compute_omega_min_q_batch` multiplies the spectral sum by `1/(2 N_k)`, adds `Delta^2/U`, and subtracts the same occupation-weighted expression evaluated at `Delta=0` and the same q.
- The k grid contains 800 points and the denominator is the configured full `N_k`.
- Temperature enters through the Fermi occupation; there is no separately coded entropy or log-partition term.
- The exact oracle separately minimizes the Delta=0 scan over q, which is zero under the same-q subtraction up to numerical precision, and compares superconducting minima against that scalar reference.

## Finding

The parent Supplement equation faithfully transcribes the production
implementation, including the full-branch sum, particle-hole factor 1/2,
pairing term, same-q normal subtraction, temperature dependence, and k-grid
normalization.  The notation nevertheless calls the expression a generic
finite-temperature free energy.  Because the implementation does not contain a
separate entropy or `-kBT log(1+exp(-E/kBT))` term, the public revision denotes
it by `F_impl` and calls it the production normal-referenced occupation-weighted
spectral objective.  No textbook formula is substituted and no frozen minimum
or label is changed.
