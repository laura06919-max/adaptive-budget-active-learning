"""Regenerate labels.csv from birdnet_filenames.npy without re-downloading audio.

If labels.csv is missing or lacks a recording_id column --> reconstructs it by loading 
only the metadata from HuggingFace BirdSet
(no audio download needed, faster than a full re-export).

Usage:
  python scripts/regenerate_labels.py --subset PER --data-dir data/birdset_per
  python scripts/regenerate_labels.py --subset POW --data-dir data/birdset_pow
  python scripts/regenerate_labels.py --subset PER --data-dir data/birdset_per --force

What it does:
  1. Loads birdnet_filenames.npy to see which files exist
  2. Loads BirdSet metadata only 
  3. Matches filenames back to dataset rows by export order
  4. Writes labels.csv with filename, label, recording_id columns
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _get_label_str(row: dict) -> str | None:
    """Extract label string from a BirdSet row."""
    label_candidates = [
        "ebird_code", "ebird_code_multilabel",
        "primary_label", "label", "species_id",
    ]
    for col in label_candidates:
        val = row.get(col)
        if val is None:
            continue
        if hasattr(val, "item"):
            try:
                val = val.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(val, (int, float)):
            return f"class_{int(val)}"
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, (list, np.ndarray)):
            arr = list(val)
            if not arr:
                continue
            first = arr[0]
            if hasattr(first, "item"):
                first = first.item()
            if isinstance(first, (int, float)):
                narr = np.asarray(arr)
                if narr.max() > 1:
                    return f"class_{int(narr[0])}"
                positives = np.where(narr.flatten() > 0)[0]
                return f"class_{int(positives[0])}" if len(positives) > 0 else f"class_{int(np.argmax(narr.flatten()))}"
            if isinstance(first, str) and first.strip():
                return first.strip()
    return None


def main(subset: str, data_dir: Path, force: bool = False) -> None:
    labels_path = data_dir / "labels.csv"
    filenames_path = data_dir / "embeddings_birdnet" / "birdnet_filenames.npy"

    # --- Check if need to regenerate ---
    if labels_path.exists() and not force:
        with open(labels_path) as f:
            header = f.readline().strip().split(",")
        if "recording_id" in header:
            print(f"labels.csv already exists with recording_id column at {labels_path}")
            print("Use --force to regenerate anyway.")
            return
        else:
            print(f"labels.csv exists but missing recording_id column — regenerating...")
    elif not labels_path.exists():
        print(f"labels.csv not found at {labels_path} — generating from scratch...")

    # --- Load filenames ---
    if not filenames_path.exists():
        raise FileNotFoundError(
            f"birdnet_filenames.npy not found at {filenames_path}. "
            "Run the embedding step first."
        )
    filenames = np.load(filenames_path, allow_pickle=True)
    n_files = len(filenames)
    print(f"Found {n_files} filenames in birdnet_filenames.npy")

    # --- Load BirdSet metadata only ---
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace datasets: pip install datasets")

    print(f"Loading BirdSet {subset} metadata (no audio — this is fast)...")

    # Load without audio column to avoid downloading audio shards
    ds = load_dataset(
        "DBD-research-group/BirdSet",
        subset,
        split="train",
        columns=[
            "filepath", "ebird_code", "ebird_code_multilabel",
            "start_time", "end_time",
        ],
    )
    print(f"Loaded {len(ds)} rows of metadata")

    # --- Match filenames to dataset rows ---
    rows = []
    n_skipped = 0
    for i in range(len(ds)):
        if len(rows) >= n_files:
            break
        row = ds[i]
        label = _get_label_str(row)
        if label is None:
            n_skipped += 1
            continue

        filepath = row.get("filepath", "")
        recording_id = Path(filepath).stem if filepath else f"unknown_{i}"

        expected_fname = f"birdset_{len(rows):05d}.wav"
        rows.append((expected_fname, label, recording_id))

    print(f"Matched {len(rows)} rows (skipped {n_skipped} with no label)")

    if len(rows) != n_files:
        print(
            f"WARNING: matched {len(rows)} rows but expected {n_files}. "
            "The dataset may have changed since export. "
            "Consider re-exporting with export_birdset_sample.py."
        )

    # --- Write labels.csv ---
    with open(labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "recording_id"])
        w.writerows(rows)

    n_classes = len(set(lbl for _, lbl, _ in rows))
    n_recordings = len(set(rid for _, _, rid in rows))
    print(f"\nWrote {labels_path}")
    print(f"  {len(rows)} samples, {n_classes} classes, {n_recordings} unique recordings")
    print(f"\nSet num_classes to {n_classes} in your experiment yaml.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate labels.csv from birdnet_filenames.npy"
    )
    parser.add_argument(
        "--subset",
        required=True,
        help="BirdSet subset name e.g. PER, HSN, POW, SSW",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Path to dataset folder e.g. data/birdset_per",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if labels.csv already exists with recording_id",
    )
    args = parser.parse_args()
    main(subset=args.subset, data_dir=args.data_dir, force=args.force)
