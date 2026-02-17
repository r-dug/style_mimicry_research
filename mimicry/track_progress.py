#!/usr/bin/env python3
"""Refresh and print pipeline progress snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.constants import DATA_ROOT_DEFAULT
from common.progress_tracker import ProgressTracker


def parse_args() -> argparse.Namespace:
    """Parse CLI args for tracker refresh."""
    parser = argparse.ArgumentParser(description="Refresh project progress tracker JSON")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT_DEFAULT,
        help="Root dataset path (default: data/style_mimicry)",
    )
    parser.add_argument(
        "--tracker-path",
        type=Path,
        default=Path("data/style_mimicry/progress/progress_tracker.json"),
        help="Progress tracker file path",
    )
    parser.add_argument(
        "--min-samples-per-artist",
        type=int,
        default=20,
        help="Minimum original samples required per artist",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print full tracker JSON to stdout",
    )
    return parser.parse_args()


def main() -> int:
    """Run a tracker refresh and print a concise summary."""
    args = parse_args()
    tracker = ProgressTracker(
        tracker_path=args.tracker_path,
        data_root=args.data_root,
        min_samples_per_artist=args.min_samples_per_artist,
    )
    state = tracker.refresh_snapshot()

    original = state["snapshot"]["original_art"]["totals"]
    protected = state["snapshot"]["protected_art"]["totals"]
    robust = state["snapshot"]["robust_samples"]["totals"]
    mimic = state["snapshot"]["mimic_art"]

    print(f"Tracker updated: {args.tracker_path}")
    print(f"Step status: {state['step_status']}")
    print(
        "Original art: "
        f"{original['artists']} artists, {original['images']} images, "
        f"{original['artists_below_threshold']} artists below threshold"
    )
    print(f"Protected art images: {protected['images']}")
    print(f"Robust sample images: {robust['images']}")
    print(f"Mimic art images: {mimic['image_count']}")

    if args.pretty:
        print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

