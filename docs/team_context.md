# Team context

This repo is a showcase of **my** contributions to the E4E active learning project. The complete pipeline (data export, embedding extraction, model heads, experiment runner, analysis) is in the private `UCSD-E4E/data_aug_active_learning` repository.

| Area | Who |
|---|---|
| Pipeline scaffold, CLI, preprocessing, mixed/ensemble and rarity strategies, GPU paths | Jeremy Frank (lead) |
| Perch/BirdNET embedding pipelines, model heads (linear, MLP, SVM, RF, XGBoost, LightGBM, prototypical), class balancing, entropy/margin/least-confidence/coreset strategies, results analysis | Shrivas Sudharsan |
| Bandit budget agent, Top-K entropy, recording-based split, QBC via MC Dropout, dynamic batch size, label regeneration, PER/UHH configs, scaled experiments and the studies in this README | Laura Yuan |

How my parts plug into the team pipeline:

- `BanditBudgetAgent` is called from the experiment runner each round (`budget_mode: bandit` in the YAML config) and logged in every round's results.
- `TopKEntropyStrategy` and `QBCStrategy` are registered in the strategy registry alongside the team's strategies.
- `recording_split` / `load_recording_ids` are used by the runner when `split_by_recording: true`.

Baseline strategies used in `demo/run_demo.py` (random, entropy, margin) are short re-implementations for the demo, not the team's code.
