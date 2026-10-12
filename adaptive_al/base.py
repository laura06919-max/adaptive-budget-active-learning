"""Minimal strategy interface (mirrors the E4E pipeline's BaseStrategy)."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseStrategy(ABC):
    """An active learning query strategy picks which unlabeled samples to label next."""

    @abstractmethod
    def select(self, model: Any, X_unlabeled: np.ndarray, budget: int, **kwargs) -> np.ndarray:
        """Return indices into X_unlabeled for the samples to acquire."""
        ...
