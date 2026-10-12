"""Multi-armed bandit for adaptive budget sizing in active learning.

bandit agent learns which budget size gives the best
accuracy gain per label spent.

Each "arm" is a budget level: [min, ..., max] in steps.
The agent tracks which arm gave the best reward historically
and shifts probability toward it using epsilon-greedy.
"""

import numpy as np


class BanditBudgetAgent:
    """Epsilon-greedy bandit for adaptive AL budget sizing.

    Args:
        min_budget:   Smallest budget to consider.
        max_budget:   Largest budget to consider.
        n_arms:       Number of discrete budget levels to try.
                      e.g. n_arms=5, min=5, max=25 gives [5, 10, 15, 20, 25]
        epsilon:      Exploration rate. 0.1 = explore 10% of rounds.
        seed:         Random seed for reproducibility.
    """

    def __init__(
        self,
        min_budget: int = 5,
        max_budget: int = 30,
        n_arms: int = 5,
        epsilon: float = 0.1,
        seed: int = 42,
        strategy: str = "epsilon_greedy",
        ucb_c: float = 2 ** 0.5,
    ) -> None:
        """
        Args:
            strategy: "epsilon_greedy" or "ucb"
        """
        self.arms = list(np.linspace(min_budget, max_budget, n_arms, dtype=int))
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Tracks reward history per arm
        self._counts = [0] * len(self.arms)
        self._values = [1.0] * len(self.arms)  #set to 1 to force exploration of all arms at start
        self.strategy = strategy
        self._total_rounds = 0
        self.ucb_c = ucb_c
        
    def select_budget(self, round_idx: int = 0) -> tuple[int, int]:
        """Choose a budget using epsilon-greedy with decay or UCB.

        Returns:
            (budget, arm_index)
        """
        self._total_rounds += 1

        if self.strategy == "ucb":
            # UCB: pick arm with highest avg_reward + exploration bonus
            # Arms never tried get infinite score so they're tried first
            scores = []
            for i in range(len(self.arms)):
                if self._counts[i] == 0:
                    scores.append(float("inf"))
                else:
                    exploit = self._values[i]
                    explore = self.ucb_c * np.sqrt(
                        np.log(self._total_rounds) / self._counts[i]
                    )
                    scores.append(exploit + explore)
            arm_idx = int(np.argmax(scores))
        else:
            # Epsilon-greedy with decay
            current_epsilon = max(0.05, self.epsilon - 0.02 * round_idx)
            if self.rng.random() < current_epsilon:
                arm_idx = int(self.rng.integers(len(self.arms)))
            else:
                arm_idx = int(np.argmax(self._values))

        return int(self.arms[arm_idx]), arm_idx

    def update(self, arm_idx: int, reward: float) -> None:
        """Update arm value with observed reward using incremental mean.

        Args:
            arm_idx: Which arm was used this round.
            reward:  Accuracy gain per label = (acc_new - acc_old) / budget.
        """
        self._counts[arm_idx] += 1
        n = self._counts[arm_idx]
        self._values[arm_idx] += (reward - self._values[arm_idx]) / n

    def state_dict(self) -> dict:
        return {
            "arms": [int(a) for a in self.arms],
            "counts": [int(c) for c in self._counts],
            "values": [round(float(v), 6) for v in self._values],
            "best_arm": int(self.arms[int(np.argmax(self._values))]),
            "epsilon": float(self.epsilon),
            "strategy": self.strategy,
            "total_rounds": self._total_rounds,
            "ucb_c": float(self.ucb_c),
        }
