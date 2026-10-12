"""Recording-aware data utilities.

Bioacoustic datasets often cut one long field recording into many short
clips. Clips from the same recording share background noise, microphone
and site, so a random clip-level train/test split can leak information:
the model recognises the recording, not the bird.

These helpers (from the E4E pipeline, feature/recording-split) load a
recording ID for every clip and split train/test by *whole recordings*.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def recording_split(
    indices: np.ndarray,
    recording_ids: np.ndarray,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by recording ID so no recording appears in both train and test.

    Groups clips by their XenoCanto recording ID, then assigns whole
    recordings to train or test. Prevents background noise leakage
    where clips from the same recording end up on both sides of the split.

    Args:
        indices:       Sample indices to split.
        recording_ids: Recording ID string for each sample, aligned to indices.
        test_frac:     Fraction of recordings to hold out as test.
        seed:          Random seed for reproducibility.

    Returns:
        (pool_indices, test_indices) — disjoint, no recording split across both.
    """
    rng = np.random.default_rng(seed)

    # Group sample indices by recording ID
    recording_to_indices: dict[str, list[int]] = {}
    for idx in indices:
        rec_id = recording_ids[idx]
        if rec_id not in recording_to_indices:
            recording_to_indices[rec_id] = []
        recording_to_indices[rec_id].append(int(idx))

 
    unique_recordings = np.array(list(recording_to_indices.keys()))
    rng.shuffle(unique_recordings)

    n_test_target = max(1, int(len(indices) * test_frac))
    test_indices = []
    pool_indices = []

    for rec_id in unique_recordings:
        clips = recording_to_indices[rec_id]
        if len(test_indices) < n_test_target:
            test_indices.extend(clips)
        else:
            pool_indices.extend(clips)

    pool_arr = np.array(pool_indices, dtype=int)
    test_arr = np.array(test_indices, dtype=int)

    logger.info(
        "Recording-based split: %d recordings → pool=%d clips, test=%d clips",
        len(unique_recordings), len(pool_arr), len(test_arr),
    )
    return pool_arr, test_arr



def load_recording_ids(config: dict[str, Any], filenames: np.ndarray) -> np.ndarray | None:
    """Load recording IDs from labels.csv if the recording_id column exists.

    Returns an array of recording ID strings aligned to the embedding order,
    or None if the column isn't present 

    Args:
        config:    Experiment config dict.
        filenames: Embedding filenames array from <encoder>_filenames.npy,
                   used to align recording IDs to embedding order.

    Returns:
        Array of shape (N,) with recording ID strings, or None.
    """
    import pandas as pd

    labels_path = Path(config["dataset"].get("labels_path", "data/labels.npy"))
    if labels_path.suffix.lower() != ".csv" or not labels_path.exists():
        return None

    df = pd.read_csv(labels_path)
    if "recording_id" not in df.columns:
        logger.info("No recording_id column found in %s — skipping recording-based split", labels_path)
        return None

    fname_to_recording = dict(zip(df["filename"], df["recording_id"]))

    missing = [fn for fn in filenames if fn not in fname_to_recording]
    if missing:
        raise ValueError(
            f"Found {len(missing)} embedding filename(s) with no recording_id in {labels_path}. "
            f"Examples: {missing[:5]}"
        )

    recording_ids = np.array([fname_to_recording[fn] for fn in filenames], dtype=str)
    logger.info("Loaded recording IDs for %d samples from %s", len(recording_ids), labels_path)
    return recording_ids
