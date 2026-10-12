"""Query by Committee strategy using Monte Carlo Dropout.

Instead of training multiple separate models (the original QBC approach),
we simulate a 'committee' by running the same MLP with dropout kept ON
during inference. Each forward pass = one committee member's vote.
The disagreement (variance) across votes = epistemic uncertainty.

"""

import numpy as np

from typing import Any as BaseModelHead  # lab pipeline type; any model with MC Dropout works
from adaptive_al.base import BaseStrategy


class QBCStrategy(BaseStrategy):
    """Query by Committee via Monte Carlo Dropout.

    For MLPHead: uses true MC Dropout epistemic uncertainty (variance
    across n_passes stochastic forward passes with dropout active).

    """

    def __init__(self, n_passes: int = 20) -> None:
        """
        Args:
            n_passes: Number of Monte Carlo forward passes.
                      20 is a good default — enough to estimate variance
                      without being too slow.
        """
        self.n_passes = n_passes

    def select(
        self,
        model: BaseModelHead,
        X_unlabeled: np.ndarray,
        budget: int,
        **kwargs,
    ) -> np.ndarray:
        """Select top-budget samples with highest epistemic uncertainty.

        Args:
            model: A trained model head (MLPHead gets true MC Dropout;
                   others get entropy fallback).
            X_unlabeled: Unlabeled embedding vectors, shape (N, D).
            budget: How many samples to select this round.

        Returns:
            Array of indices into X_unlabeled, sorted by decreasing
            epistemic uncertainty. Length = min(budget, N).
        """
        n = len(X_unlabeled)
        budget = min(budget, n)

        # --- Path 1: MLPHead with MC Dropout ---
        if hasattr(model, "predict_proba_mc_variance"):
            print(
                f"QBC: using MC Dropout with {self.n_passes} passes "
                f"on {n} unlabeled samples"
            )
            uncertainty = model.predict_proba_mc_variance(
                X_unlabeled,
                n_passes=self.n_passes,
            )
            top_indices = np.argsort(-uncertainty)[:budget]
            print(
                f"QBC: selected {budget} samples. "
                f"Uncertainty range: "
                f"min={float(np.min(uncertainty)):.4f}, "
                f"max={float(np.max(uncertainty)):.4f}, "
                f"mean={float(np.mean(uncertainty)):.4f}"
            )
            return top_indices

        raise TypeError(
    f"QBCStrategy requires a model with MC Dropout support (e.g. MLPHead), "
    f"but got '{type(model).__name__}'. Use a different strategy (e.g. entropy) "
    f"for this model type."
)