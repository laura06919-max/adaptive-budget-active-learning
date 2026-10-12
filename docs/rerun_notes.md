# October 2026 re-run: methodology

Before publishing, I audited my original May 2026 results against the logs and re-ran the key experiments. The numbers in the README come from this re-run.

## What was wrong in the original runs

1. **Unequal comparison.** The headline "UCB 81% vs fixed 68%" compared a 20-round bandit run (≈340 training labels) with an 8-round fixed run (145 labels).
2. **Bandit credit assignment.** The reward for the arm chosen in round *r* was computed from the accuracy change produced by round *r−1*'s batch, and it used **test** accuracy.
3. **Reward scale.** Raw rewards (~0.003) were ~300× smaller than UCB's exploration bonus, so UCB pulled every arm exactly 4 times in 20 rounds.
4. **Lost code.** When I rebuilt the bandit branch on top of `main`, the recording-split code and the `recording_ids` hand-off to Top-K Entropy were left out, so the config flags had no effect.
5. **Single seed.** All runs used seed 42.

## What the re-run changed

- Restored the recording-based split and passed `recording_ids` to strategies.
- Bandit reward = 100 × (validation-accuracy gain of the batch) / labels in that batch, credited to the arm that chose that batch. The original behaviour was kept behind a flag and re-run for comparison ("UCB original").
- UCB exploration constant `ucb_c` made configurable (tested 1.41, 0.5, 0.1; the README reports 1.41, the textbook default; the others were within noise).
- Fixed-budget baselines at 5, 15 and 17 clips/round; comparisons made at **matched label counts** by interpolating each learning curve.
- 3 seeds (42, 123, 456) for every condition.

## Findings about the data

- In the exported PER sample, each of the 2,000 clips comes from a **different** recording (same for UHH, 500/500). Cross-recording leakage is therefore impossible in this sample, the recording split acts like an unstratified random split, and Top-K Entropy reduces exactly to entropy (identical curves in every seed).
- The drop in the full-data "ceiling" from ~85% to ~79% that I'd attributed to leakage is explained by stratified vs unstratified splitting (82.9% vs 79.2% in the re-run).
- AL with 348 labels slightly exceeds the full-data random forest ceiling. I report this as an observation, not a claim: the ceiling is a single fixed RF configuration on an imbalanced training set.

## Environment

Re-run with Python 3.10, scikit-learn 1.7.2, NumPy 1.26. The original runs used scikit-learn 1.8 on macOS. The identical config differed by ~4 points between the two environments, which is part of why only multi-seed means are reported.

One class-balanced run (seed 123) crashed on a pre-existing bug in the team's class-balanced strategy (it fails when a species is missing from the labeled set), so that condition has 2 seeds.
