"""Dynamic budget computation for Active Learning.

Instead of a fixed budget_per_round, computes how many unlabeled
samples the model is still uncertain about each round and uses that
as the batch size. Inspired by Adachi et al., "Adaptive Batch Sizes
for Active Learning" (AISTATS 2024).
"""

import numpy as np


def compute_dynamic_budget(
    model,
    X_unlabeled: np.ndarray,
    entropy_threshold: float = 0.5,
    min_budget: int = 5,
    max_budget: int = 60,
    fallback_budget: int = 15,
) -> tuple[int, dict]:
    """Compute this round's batch size based on model uncertainty.

    Counts how many unlabeled samples have entropy ABOVE the threshold.
    That count is the budget — clamped between min and max.

    Args:
        model: Any trained model head with predict_proba() method.
        X_unlabeled: Unlabeled embeddings, shape (N, D).
        entropy_threshold: Samples above this entropy are "uncertain".
                           Range is 0.0 to log(n_classes).
                           0.5 is a reasonable default for 5 classes.
        min_budget: Never label fewer than this many samples per round.
        max_budget: Never label more than this many samples per round.
        fallback_budget: Used if model has no predict_proba (safety net).

    Returns:
        Tuple of:
          - budget (int): how many samples to label this round
          - stats (dict): diagnostic info for logging/results
    """
    # Safety net — if model can't give probabilities, use fallback
    if not hasattr(model, "predict_proba"):
        print(
            f"DynamicBudget: model has no predict_proba, "
            f"using fallback budget {fallback_budget}"
        )
        return fallback_budget, {"method": "fallback"}

    # Get probability distribution over classes for every unlabeled sample
    probs = model.predict_proba(X_unlabeled)          # shape (N, C)
    probs = np.clip(probs, 1e-12, 1.0)                # avoid log(0)

    entropy = -np.sum(probs * np.log(probs), axis=1)  # shape (N,)

    # Count how many samples are "uncertain enough" to be worth labeling
    uncertain_mask = entropy > entropy_threshold
    n_uncertain = int(np.sum(uncertain_mask))

    # Clamp between min and max so we never do something extreme
    budget = int(np.clip(n_uncertain, min_budget, max_budget))

    stats = {
        "method": "dynamic_entropy",
        "entropy_threshold": entropy_threshold,
        "n_unlabeled": len(X_unlabeled),
        "n_uncertain": n_uncertain,
        "entropy_mean": float(np.mean(entropy)),
        "entropy_max": float(np.max(entropy)),
        "budget_before_clamp": n_uncertain,
        "budget_after_clamp": budget,
        "min_budget": min_budget,
        "max_budget": max_budget,
    }

    print(
        f"DynamicBudget: {n_uncertain}/{len(X_unlabeled)} samples above "
        f"threshold {entropy_threshold:.2f} → budget={budget} "
        f"(entropy mean={stats['entropy_mean']:.3f})"
    )

    return budget, stats