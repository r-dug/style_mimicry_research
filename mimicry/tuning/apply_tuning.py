#!/usr/bin/env python3
"""Step 3: Fine-tune SD3.5 Large with DreamBooth LoRA, per-artist, with bookmark-aware resuming."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICRY_DIR = REPO_ROOT / "mimicry"
if str(MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICRY_DIR))

from common.constants import (  # noqa: E402
    DEFAULT_LORA_OUTPUT_DIR,
    DEFAULT_SD35_MODEL,
    IMAGE_EXTENSIONS,
    PROTECTED_METHODS,
    ROBUST_METHODS,
    TUNING_STACKS,
)
from common.progress_tracker import ProgressTracker  # noqa: E402
from experiment_config import get_pilot_artists_by_style  # noqa: E402

STEP_NAME = "step_3_fine_tune_sd35"
IMAGE_EXTENSIONS_SET = set(IMAGE_EXTENSIONS)

# Wrapper script filename per stack — each stack owns its own hyperparameter defaults.
WRAPPER_NAMES: dict[str, str] = {
    "hf_diffusers": "local_dreambooth_lora_sd3_wrapper.py",
    "simpletuner": "local_simpletuner_wrapper.py",
}


@dataclass(frozen=True)
class TuningTarget:
    """One artist directory target for fine-tuning."""

    split: str
    style: str
    artist: str
    source_dir: Path
    output_dir: Path
    protection: str
    countermeasure: str


def parse_args() -> argparse.Namespace:
    """Parse CLI args for Step 3 fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SD3.5 Large with DreamBooth LoRA per artist.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/style_mimicry"),
        help="Dataset root path.",
    )
    parser.add_argument(
        "--stack",
        default="hf_diffusers",
        choices=TUNING_STACKS,
        help="Training stack to use.",
    )
    parser.add_argument(
        "--protection",
        default="none",
        help="Protection method applied to training data. 'none' uses original art.",
    )
    parser.add_argument(
        "--countermeasure",
        default="none",
        help="Countermeasure applied to training data. 'none' uses unprocessed images.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=["historical"],
        help="Dataset split(s) to process (repeatable).",
    )
    parser.add_argument("--style", action="append", default=[], help="Style filter (repeatable).")
    parser.add_argument("--artist", action="append", default=[], help="Artist filter (repeatable).")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Train on one pilot artist per style only.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Retrain even if output exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument(
        "--no-push-to-hub",
        action="store_true",
        help="Disable HuggingFace Hub upload.",
    )
    parser.add_argument(
        "--min-samples-per-artist",
        type=int,
        default=20,
        help="Minimum images required to train on an artist.",
    )

    # Hyperparameter overrides are forwarded verbatim to the wrapper via extra_args,
    # keeping defaults in one place (the wrapper) rather than duplicated here.
    return parser.parse_known_args()


def resolve_source_dir(
    data_root: Path,
    split: str,
    style: str,
    artist: str,
    protection: str,
    countermeasure: str,
) -> Path:
    """Resolve the input image directory based on protection/countermeasure flags."""
    if countermeasure != "none":
        return data_root / "robust_samples" / countermeasure / split / style / artist
    if protection != "none":
        return data_root / "protected_art" / protection / split / style / artist
    return data_root / "original_art" / split / style / artist


def image_count(directory: Path) -> int:
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS_SET)


def discover_tuning_targets(
    data_root: Path,
    splits: list[str],
    protection: str,
    countermeasure: str,
    stack: str,
    style_filter: set[str],
    artist_filter: set[str],
    pilot_artists: dict[str, str] | None,
    min_samples: int,
) -> list[TuningTarget]:
    """Discover artist targets for fine-tuning."""
    targets: list[TuningTarget] = []
    variant = f"{protection}__{countermeasure}"
    output_base = DEFAULT_LORA_OUTPUT_DIR / stack / variant

    # Walk the original_art tree for the canonical artist list
    for split in splits:
        original_split = data_root / "original_art" / split
        if not original_split.exists():
            continue
        for style_dir in sorted(p for p in original_split.iterdir() if p.is_dir()):
            style = style_dir.name
            if style_filter and style not in style_filter:
                continue
            for artist_dir in sorted(p for p in style_dir.iterdir() if p.is_dir()):
                artist = artist_dir.name
                if artist_filter and artist not in artist_filter:
                    continue
                if pilot_artists is not None and pilot_artists.get(style) != artist:
                    continue

                source = resolve_source_dir(data_root, split, style, artist, protection, countermeasure)
                count = image_count(source)
                if count < min_samples:
                    print(f"  skip {split}/{style}/{artist}: {count} images < {min_samples} minimum")
                    continue

                targets.append(TuningTarget(
                    split=split,
                    style=style,
                    artist=artist,
                    source_dir=source,
                    output_dir=output_base / split / style / artist,
                    protection=protection,
                    countermeasure=countermeasure,
                ))
    return targets


def run_training(
    target: TuningTarget,
    args: argparse.Namespace,
    extra_args: list[str],
) -> dict[str, str | int]:
    """Invoke the wrapper for one artist and return a result summary."""
    wrapper_name = WRAPPER_NAMES.get(args.stack)
    if not wrapper_name:
        return {"status": "failed_setup", "error": f"no wrapper registered for stack '{args.stack}'"}
    wrapper = MIMICRY_DIR / "tuning" / args.stack / wrapper_name
    if not wrapper.exists():
        return {"status": "failed_setup", "error": f"wrapper not found: {wrapper}"}

    cmd = [
        "python3", str(wrapper),
        "--input-dir", str(target.source_dir),
        "--output-dir", str(target.output_dir),
        "--protection", target.protection,
        "--countermeasure", target.countermeasure,
        *extra_args,
    ]

    if args.no_push_to_hub:
        cmd.append("--no-push-to-hub")

    if args.dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return {"status": "complete", "dry_run": True}

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as error:
        return {"status": "failed", "exit_code": error.returncode}

    # Verify output
    lora_weights = target.output_dir / "pytorch_lora_weights.safetensors"
    if lora_weights.exists():
        return {"status": "complete"}
    return {"status": "incomplete", "error": "lora weights not found after training"}


def main() -> int:
    """Run Step 3 fine-tuning and update progress tracker."""
    args, extra_args = parse_args()

    # Validate protection/countermeasure names
    if args.protection != "none" and args.protection not in PROTECTED_METHODS:
        print(f"Unknown protection: {args.protection}. Options: none, {', '.join(PROTECTED_METHODS)}")
        return 2
    if args.countermeasure != "none" and args.countermeasure not in ROBUST_METHODS:
        print(f"Unknown countermeasure: {args.countermeasure}. Options: none, {', '.join(ROBUST_METHODS)}")
        return 2

    # Pilot mode
    pilot_artists = None
    if args.pilot:
        pilot_artists = get_pilot_artists_by_style(args.data_root, split=args.split[0])
        if not pilot_artists:
            print("No pilot artists found.")
            return 2
        print(f"Pilot mode: {pilot_artists}")

    targets = discover_tuning_targets(
        data_root=args.data_root,
        splits=sorted(dict.fromkeys(args.split)),
        protection=args.protection,
        countermeasure=args.countermeasure,
        stack=args.stack,
        style_filter=set(args.style),
        artist_filter=set(args.artist),
        pilot_artists=pilot_artists,
        min_samples=args.min_samples_per_artist,
    )

    if not targets:
        print("No tuning targets found.")
        return 0

    tracker = ProgressTracker(
        tracker_path=args.data_root / "progress" / "progress_tracker.json",
        data_root=args.data_root,
        min_samples_per_artist=args.min_samples_per_artist,
    )
    tracker.load()

    variant_label = f"{args.protection}/{args.countermeasure}"
    print(
        f"Fine-tuning {len(targets)} artist(s) with stack={args.stack}, "
        f"variant={variant_label}, dry_run={args.dry_run}"
    )

    completed = 0
    failed = 0
    skipped = 0

    for target in targets:
        bookmark_key = (
            f"{args.stack}/{target.protection}/{target.countermeasure}"
            f"/{target.split}/{target.style}/{target.artist}"
        )
        label = f"{target.split}/{target.style}/{target.artist}"

        # Check if already complete
        if not args.overwrite:
            lora_weights = target.output_dir / "pytorch_lora_weights.safetensors"
            if lora_weights.exists():
                print(f"  skip (complete): {label}")
                skipped += 1
                continue

        print(f"  training: {label} [{image_count(target.source_dir)} images]")
        result = run_training(target, args, extra_args)

        if not args.dry_run:
            tracker.set_bookmark(
                step_name=STEP_NAME,
                bookmark_key=bookmark_key,
                payload={
                    "stack": args.stack,
                    "protection": target.protection,
                    "countermeasure": target.countermeasure,
                    "split": target.split,
                    "style": target.style,
                    "artist": target.artist,
                    **result,
                },
            )

        if result.get("status") == "complete":
            completed += 1
        else:
            failed += 1
            print(f"  FAILED: {label} -> {result}")

    print(f"\nSummary: {completed} completed, {skipped} skipped, {failed} failed")

    if not args.dry_run:
        tracker.refresh_snapshot()
        print(f"Updated tracker: {args.data_root / 'progress' / 'progress_tracker.json'}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
