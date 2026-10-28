# Descriptor semantics audit

The source fields were verified against the exact-oracle and Stage V-v4
descriptor implementation. `free_energy_gap_to_normal` is
\(F_{SC}^{min}-F_{normal}\); a superconducting label requires a positive-Delta
candidate and a negative free-energy gain beyond tolerance. Phase and topology
labels are evaluation-only. Normal-state q has `q_status=not_applicable` and is
masked, never treated as a physical zero. Topology/gap/Pfaffian outputs require
`topology_trusted`; non-finite, unresolved, or untrusted values are masked.
Proposal, action, reward, iteration, graph-node, and candidate-node fields do
not enter the descriptor. The common final schema lacks reliable second-basin
fields, so the landscape-summary ablation is unavailable rather than imputed.
