"""Top-K Entropy strategy for active learning.

Groups unlabeled clips by their source recording ID and scores each
recording by the mean entropy of its top-K most uncertain clips.

This avoids wasting the labeling budget on multiple clips from the same
recording (which share background noise and give redundant information).

Falls back to standard clip-level entropy when recording_ids are not
available (e.g. datasets without recording ID in labels.csv).

Reference: Martinsson et al. (2025) - Top K entropy achieves comparable
performance to fully labeled data with only 8% of labels on soundscape
bioacoustic data.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from adaptive_al.base import BaseStrategy


class TopKEntropyStrategy(BaseStrategy):
    """Recording-aware entropy sampling.

    Scores each recording by the mean entropy of its top-K most
    uncertain clips, then selects the most uncertain recordings.

    Args:
        top_k: Number of clips per recording to average entropy over.
               Defaults to 3. If a recording has fewer clips, all are used.
    """

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def select(
        self,
        model: Any,
        X_unlabeled: np.ndarray,
        budget: int,
        seed: int,
        recording_ids: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Select clips using recording-aware entropy.

        Args:
            model:         Trained model with predict_proba method.
            X_unlabeled:   Embeddings of unlabeled clips (N, D).
            budget:        Number of clips to select.
            seed:          Random seed for tie-breaking.
            recording_ids: Array of recording IDs aligned to X_unlabeled.
                           If None, falls back to standard entropy.

        Returns:
            Indices into X_unlabeled of selected clips.
        """
        n = len(X_unlabeled)
        budget = int(min(max(1, budget), n))

        probs = model.predict_proba(X_unlabeled)
        probs = np.clip(probs, 1e-10, 1.0)

        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=1)

        # Normalize rows
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(row_sums > 0, row_sums, 1.0)

        # Shannon entropy per clip
        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        # If no recording IDs, fall back to standard entropy
        if recording_ids is None:
            top_indices = np.argsort(entropies)[::-1][:budget]
            return top_indices

        # --- Recording-aware selection ---
        recording_to_clips: dict[str, list[int]] = {}
        for i, rec_id in enumerate(recording_ids):
            key = str(rec_id)
            if key not in recording_to_clips:
                recording_to_clips[key] = []
            recording_to_clips[key].append(i)

        # Score each recording by mean entropy of top-K clips
        recording_scores: list[tuple[float, str]] = []
        for rec_id, clip_indices in recording_to_clips.items():
            clip_entropies = entropies[clip_indices]
            # Take top-K most uncertain clips
            top_k = min(self.top_k, len(clip_entropies))
            top_k_entropies = np.sort(clip_entropies)[::-1][:top_k]
            score = float(top_k_entropies.mean())
            recording_scores.append((score, rec_id))

      
        recording_scores.sort(key=lambda x: x[0], reverse=True)

        # Select clips from top recordings until budget is filled
        selected: list[int] = []
        for _, rec_id in recording_scores:
            if len(selected) >= budget:
                break
            clip_indices = recording_to_clips[rec_id]
            # From this recording, pick the most uncertain clip
            clip_entropies = entropies[clip_indices]
            best_clip_local = int(np.argmax(clip_entropies))
            selected.append(clip_indices[best_clip_local])

        # safety fallback
        if len(selected) < budget:
            remaining = np.setdiff1d(np.arange(n), selected)
            extra = np.argsort(entropies[remaining])[::-1][:budget - len(selected)]
            selected.extend(remaining[extra].tolist())

        return np.array(selected[:budget], dtype=int)
