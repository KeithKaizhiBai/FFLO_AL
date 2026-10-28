# Exact-count semantics audit

## Status

`PASS`

The counts refer to different run stages and data layers; they are not
interchangeable aliases.

| Quantity | Meaning |
|---|---|
| 8,192 | Stage-I selection-history records, including repeated coordinates |
| 7,305 | Unique Stage-I selected coordinates |
| 6,880 | Unique coordinates retained in the Stage-I exact endpoint; numerator of the public 8.89% headline |
| 6,104 | Trusted subset of that Stage-I endpoint |
| 7,434 | Unique coordinates in the Stage-II endpoint used by Fig. 2 panel (c) |

The Stage-II endpoint contains all 6,880 Stage-I endpoint coordinates plus 554
new exact coordinates.  All shared coordinates retain the same phase label.
The combined Stage-I/II selection history contains 9,216 records and 7,885
unique selected coordinates; 7,434 appear in the Stage-II endpoint.  Thus Fig.
2's 7,434-point text is correct for the Stage-II reliability endpoint, while
the 6,880/77,361 efficiency statement is correct only for the Stage-I discovery
endpoint.  The revised caption and Supplement label both stages explicitly.

Source hashes:

- Stage I: `2e9ac68e4d891390c78c772b3d57ef8b55b69a82881adc956fa2fb17c2c76cbe`
- Stage II: `f2d751d0cfae8a3c63da786109903ffd1969a3f90c20a17d92665d8f252e2700`
