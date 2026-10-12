"""Runnable demo on synthetic "bioacoustic-like" embeddings (no lab data needed).

Each synthetic species has several "recordings"; every recording contributes
several clips that share a recording-specific offset (think: background noise
and microphone). That structure lets the demo show three things:

  1. Why the recording-based split matters (clip-level splits look too good).
  2. Uncertainty sampling vs. random, plus the recording-aware Top-K Entropy.
  3. A fixed budget vs. the UCB bandit choosing the budget each round.

Run from the repo root:
    python demo/run_demo.py            # ~1-2 minutes on a laptop
    python demo/run_demo.py --plot     # also saves demo/demo_curves.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_al import BanditBudgetAgent, TopKEntropyStrategy, recording_split  # noqa: E402
from adaptive_al.loop import make_model, run_active_learning  # noqa: E402


# ---------------------------------------------------------------- synthetic data
def make_dataset(n_species=10, recs_per_species=12, clips=(3, 9), dim=64, seed=0):
    """Clip embedding = species signal + recording offset + clip noise."""
    rng = np.random.default_rng(seed)
    species_means = rng.normal(0, 1.0, (n_species, dim))
    X, y, rec = [], [], []
    for s in range(n_species):
        for r in range(recs_per_species):
            offset = rng.normal(0, 1.4, dim)  # shared "background" for this recording
            for _ in range(rng.integers(*clips)):
                X.append(species_means[s] + offset + rng.normal(0, 0.8, dim))
                y.append(s)
                rec.append(f"rec_{s:02d}_{r:02d}")
    return np.array(X), np.array(y), np.array(rec)


# ---------------------------------------------------------------- simple baselines
def entropy_of(probs):
    p = np.clip(probs, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def select_random(model, X_unl, budget, seed=0, **_):
    return np.random.default_rng(seed).choice(len(X_unl), size=budget, replace=False)


def select_entropy(model, X_unl, budget, **_):
    return np.argsort(-entropy_of(model.predict_proba(X_unl)))[:budget]


def select_margin(model, X_unl, budget, **_):
    p = np.sort(model.predict_proba(X_unl), axis=1)
    return np.argsort(p[:, -1] - p[:, -2])[:budget]


_topk = TopKEntropyStrategy(top_k=3)


def select_top_k_entropy(model, X_unl, budget, seed=0, recording_ids=None, **_):
    return _topk.select(model, X_unl, budget, seed=seed, recording_ids=recording_ids)


# ---------------------------------------------------------------- helpers
def clip_split(n, frac, seed):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * frac)
    return idx[k:], idx[:k]


def acc_at(history, n_labels):
    """Test accuracy interpolated at a given number of labels (fair comparison)."""
    xs = [h["n_labeled"] for h in history]
    ys = [h["test_acc"] for h in history]
    return float(np.interp(n_labels, xs, ys)) if xs[0] <= n_labels <= xs[-1] else float("nan")


def fmt(values):
    v = [x for x in values if not np.isnan(x)]
    if len(v) != len(values):
        return "     n/a    "
    return f"{np.mean(v)*100:5.1f} ± {np.std(v, ddof=1)*100:3.1f}%"


def main(plot: bool = False, seeds=(0, 1, 2, 3, 4)) -> None:
    t0 = time.time()
    X, y, rec = make_dataset()
    print(f"Synthetic data: {len(X)} clips, {len(set(rec))} recordings, {len(set(y))} species\n")

    # 1. Leakage --------------------------------------------------------------
    print("1) Train on everything except the test set — clip split vs recording split")
    for name in ["clip-level split", "recording split"]:
        accs = []
        for s in seeds:
            if name == "clip-level split":
                tr, te = clip_split(len(X), 0.2, s)
            else:
                tr, te = recording_split(np.arange(len(X)), rec, test_frac=0.2, seed=s)
            m = make_model(s).fit(X[tr], y[tr])
            accs.append(np.mean(m.predict(X[te]) == y[te]))
        print(f"   {name:<18} accuracy = {fmt(accs)}")
    print("   -> the clip-level number is inflated: test clips share recordings with training clips.\n")

    # Shared recording-based split for the AL experiments
    curves: dict[str, list] = {}

    def experiment(label, select, bandit_factory=None, budget=15):
        runs = []
        for s in seeds:
            pool, test = recording_split(np.arange(len(X)), rec, test_frac=0.2, seed=s)
            pool, val = recording_split(pool, rec, test_frac=0.15, seed=s + 100)
            bandit = bandit_factory(s) if bandit_factory else None
            runs.append(run_active_learning(X, y, pool, val, test, select, n_initial=20, rounds=14,
                                            budget=budget, bandit=bandit, recording_ids=rec, seed=s))
        curves[label] = runs
        return runs

    # 2. Strategies ------------------------------------------------------------
    print("2) Query strategies (fixed budget 15/round, recording split, mean ± SD over 5 seeds)")
    for label, fn in [("random", select_random), ("entropy", select_entropy),
                      ("margin", select_margin), ("top-k entropy", select_top_k_entropy)]:
        runs = experiment(label, fn)
        print(f"   {label:<14} accuracy @ 200 labels = {fmt([acc_at(h, 200) for h in runs])}")
    print()

    # 3. Budget policy ---------------------------------------------------------
    print("3) Budget policy with entropy sampling (compare at the SAME number of labels)")
    experiment("fixed 5", select_entropy, budget=5)
    experiment("fixed 30", select_entropy, budget=30)
    experiment("UCB bandit", select_entropy,
               bandit_factory=lambda s: BanditBudgetAgent(5, 30, 5, seed=s, strategy="ucb", ucb_c=1.41))
    for label in ["fixed 5", "entropy", "fixed 30", "UCB bandit"]:
        name = "fixed 15" if label == "entropy" else label
        print(f"   {name:<11} accuracy @ 80 labels = {fmt([acc_at(h, 80) for h in curves[label]])}"
              f"   @ 200 labels = {fmt([acc_at(h, 200) for h in curves[label]])}")
    print("   ('n/a' = that policy never reached that many labels in 14 rounds)")
    print("   Note: differences smaller than ~1 SD are noise. Results shift slightly across")
    print("   machines/library versions because near-tied picks can flip and compound over rounds.")
    print(f"\nDone in {time.time() - t0:.0f}s.")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for label, color in zip(["margin", "entropy", "top-k entropy", "random"],
                                ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]):
            grid = np.linspace(20, 200, 40)
            ys = np.mean([[acc_at(h, g) for g in grid] for h in curves[label]], axis=0) * 100
            ax.plot(grid, ys, lw=2, color=color, label=label)
        ax.set_xlabel("Labeled clips")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Demo: synthetic data, recording split, mean of 5 seeds", loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df")
        ax.legend(frameon=False)
        out = Path(__file__).resolve().parent / "demo_curves.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="save demo/demo_curves.png")
    main(plot=ap.parse_args().plot)
