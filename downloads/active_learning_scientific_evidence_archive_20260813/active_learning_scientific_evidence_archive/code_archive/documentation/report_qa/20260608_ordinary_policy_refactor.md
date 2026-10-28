# Ordinary Branch Policy Notes

Date: 2026-06-08

## Question

Why can energy-window pruning appear ineffective even after rank-and-cap?

## Answer

Energy-window pruning is intentionally ordinary-only.  It must not remove
selected global-best, edge-risk, Delta-near-epsilon, or near-degenerate basins,
because those branches protect ambiguity and coverage guardrails.

Therefore, energy-window pruning only affects ordinary optional basins.  If the
ranked mandatory set already fills the total target cap, the energy window can
correctly prune ordinary candidates while having no effect on the final selected
target count.

## Report Use

When describing this stage, do not present energy-window pruning as the primary
runtime fix.  The primary fix for target explosion is mandatory rank-and-cap.
Energy-window pruning is a secondary ordinary-branch policy and should be
reported through ordinary counts before and after pruning.
