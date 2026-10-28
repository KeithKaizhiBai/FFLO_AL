# Figure 2 scientific validation

- Frozen endpoint: `dataset_iter035.npz`
- Exact rows: **7434**
- Phase counts: Normal **1867**, Uniform SC **715**, FFLO **4852**
- Cold start: **512** selected; **506** have frozen Stage I exact labels and are retained in the Stage II endpoint
- Selection history: iterations **0-34**, **9216** selected calls
- Hard-risk audit layer: **129** archived exact calls, encoded separately from phase; **19** coordinate matches remain in the final trusted endpoint
- Final phase-map change: **0.001628728** (threshold 0.002000)
- Final Normal-SC shift: **0.004166667** (threshold 0.004166667)
- Final Uniform-SC-FFLO shift: **0.000000000**
- Final trusted surprise: **0.000000000** (threshold 0.050000)

No interpolation, smoothing, gridding, resampling, boundary extraction, or manual label change was performed. The hero panel is an exact-sample map; no new smooth boundary is claimed. The six cold-start selections without a retained Stage I exact result remain visible as unmatched calls rather than receiving inferred phase labels.

`GATE_FIG2_HUMAN_SCIENCE_REVIEW = HUMAN_WAIT`
