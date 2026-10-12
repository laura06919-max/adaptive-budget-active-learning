"""A compact, self-contained active learning loop.

This is a standalone re-implementation written for this showcase repo so the
components can be run without the full (private) E4E pipeline. It keeps the
same structure as the lab pipeline:

    train -> evaluate -> (bandit picks budget) -> strategy picks samples -> repeat

and uses the corrected bandit credit assignment found during the October 2026
re-run: the reward for the batch acquired in round r is measured on the
*validation* set after retraining in round r+1 (never on the test set).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from adaptive_al.bandit_budget import BanditBudgetAgent


def make_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1)


def predict_proba_full(model: RandomForestClassifier, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba padded to all classes (a class may be missing from the labeled set)."""
    p = model.predict_proba(X)
    full = np.zeros((len(X), n_classes))
    full[:, model.classes_] = p
    return full


class _PaddedModel:
    """Wraps a fitted model so strategies always see all n_classes columns."""

    def __init__(self, model, n_classes: int) -> None:
        self.model, self.n_classes = model, n_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return predict_proba_full(self.model, X, self.n_classes)


def run_active_learning(
    X: np.ndarray,
    y: np.ndarray,
    pool_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    select: Callable[..., np.ndarray],
    n_initial: int = 20,
    rounds: int = 12,
    budget: int = 15,
    bandit: BanditBudgetAgent | None = None,
    recording_ids: np.ndarray | None = None,
    reward_scale: float = 100.0,
    seed: int = 0,
) -> list[dict]:
    """Run one active learning experiment and return one record per round.

    Args:
        select: function(model, X_unlabeled, budget, seed, recording_ids) -> indices.
        bandit: if given, the bandit chooses the budget each round; otherwise fixed `budget`.
        reward_scale: bandit reward = reward_scale * (val-accuracy gain) / labels acquired.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    labeled = rng.choice(pool_idx, size=n_initial, replace=False)
    unlabeled = np.setdiff1d(pool_idx, labeled)
    history, pending = [], None  # pending = (arm, n_acquired, val_acc_before)

    for r in range(rounds):
        model = make_model(seed).fit(X[labeled], y[labeled])
        test_acc = float(np.mean(model.predict(X[test_idx]) == y[test_idx]))
        val_acc = float(np.mean(model.predict(X[val_idx]) == y[val_idx]))

        # Credit the PREVIOUS round's arm with the gain its batch produced.
        if bandit is not None and pending is not None:
            arm, n_acq, val_before = pending
            bandit.update(arm, reward_scale * (val_acc - val_before) / n_acq)
            pending = None

        history.append({"round": r + 1, "n_labeled": int(len(labeled)), "test_acc": test_acc, "val_acc": val_acc})
        if len(unlabeled) == 0:
            break

        if bandit is not None:
            b, arm = bandit.select_budget(round_idx=r)
        else:
            b, arm = budget, None
        b = min(b, len(unlabeled))

        rec = recording_ids[unlabeled] if recording_ids is not None else None
        picked = select(_PaddedModel(model, n_classes), X[unlabeled], b, seed=seed + r, recording_ids=rec)
        new = unlabeled[picked]
        labeled = np.concatenate([labeled, new])
        unlabeled = np.delete(unlabeled, picked)
        history[-1]["budget"] = int(b)

        if bandit is not None:
            pending = (arm, len(new), val_acc)

    return history
