#!/usr/bin/env python3
"""Apply protection methods to original art samples with bookmark-aware resuming."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICRY_DIR = REPO_ROOT / "mimicry"
if str(MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICRY_DIR))

from common.constants import IMAGE_EXTENSIONS  # noqa: E402
from common.image_validation import remove_corrupt_images  # noqa: E402
from common.progress_tracker import ProgressTracker  # noqa: E402
from protections.catalog import get_method_list  # noqa: E402


@dataclass(frozen=True)
class ArtistTarget:
    """One artist directory target under a dataset split and style."""

    split: str
    style: str
    artist: str
    source_dir: Path
    output_root: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI args for Step 2 protection application."""
    parser = argparse.ArgumentParser(description="Apply perturbation protections to original art samples.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/style_mimicry"),
        help="Dataset root path.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Protection method(s) to run (repeatable). Defaults to all.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=["historical"],
        help="Dataset split(s) to process (repeatable). Default: historical.",
    )
    parser.add_argument(
        "--style",
        action="append",
        default=[],
        help="Style folder filter (repeatable).",
    )
    parser.add_argument(
        "--artist",
        action="append",
        default=[],
        help="Artist folder filter (repeatable).",
    )
    parser.add_argument(
        "--limit-artists",
        type=int,
        default=0,
        help="Maximum number of artists to process per method (0 means no limit).",
    )
    parser.add_argument(
        "--mode",
        choices=("external", "external_batch_artist", "placeholder_copy"),
        default="external_batch_artist",
        help="Execution mode for protection application.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even when they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions without writing outputs.",
    )
    parser.add_argument(
        "--min-samples-per-artist",
        type=int,
        default=20,
        help="Threshold used by tracker metadata/snapshots.",
    )
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    """Return sorted image files for one directory."""
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_targets(
    data_root: Path,
    splits: list[str],
    style_filter: set[str],
    artist_filter: set[str],
) -> list[ArtistTarget]:
    """Discover artist targets from original_art tree."""
    targets: list[ArtistTarget] = []
    base_original = data_root / "original_art"
    base_protected = data_root / "protected_art"

    for split in splits:
        split_dir = base_original / split
        if not split_dir.exists():
            continue
        for style_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            style = style_dir.name
            if style_filter and style not in style_filter:
                continue
            for artist_dir in sorted(path for path in style_dir.iterdir() if path.is_dir()):
                artist = artist_dir.name
                if artist_filter and artist not in artist_filter:
                    continue
                targets.append(
                    ArtistTarget(
                        split=split,
                        style=style,
                        artist=artist,
                        source_dir=artist_dir,
                        output_root=base_protected,
                    )
                )
    return targets


def run_external_method(method: str, input_path: Path, output_path: Path, dry_run: bool) -> None:
    """Call method-specific external runner script for one image."""
    runner = REPO_ROOT / "mimicry" / "protections" / method / "run_protection.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing external runner: {runner}")

    cmd = [
        "python3",
        str(runner),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True)


def run_external_method_batch(method: str, input_dir: Path, output_dir: Path, dry_run: bool) -> None:
    """Call method-specific external runner script for one artist directory."""
    runner = REPO_ROOT / "mimicry" / "protections" / method / "run_protection.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing external runner: {runner}")

    cmd = [
        "python3",
        str(runner),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True)


def get_external_runner(method: str) -> Path:
    """Return external runner path for a protection method."""
    return REPO_ROOT / "mimicry" / "protections" / method / "run_protection.py"


def check_external_runner(method: str, runner_path: Path) -> tuple[bool, str]:
    """Run one-time preflight check for a method external runner."""
    if not runner_path.exists():
        return False, f"missing_runner:{runner_path}"

    command = ["python3", str(runner_path), "--check-only"]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        if not details:
            details = f"runner_preflight_failed:{runner_path}"
        return False, details
    return True, ""


def apply_one_image(
    method: str,
    source_image: Path,
    output_image: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> str:
    """Apply one protection transformation and return operation status."""
    if output_image.exists() and not overwrite:
        return "skipped_existing"

    if mode == "placeholder_copy":
        if dry_run:
            print(f"[dry-run] copy {source_image} -> {output_image}")
            return "processed"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, output_image)
        return "processed"

    output_image.parent.mkdir(parents=True, exist_ok=True)
    run_external_method(method=method, input_path=source_image, output_path=output_image, dry_run=dry_run)
    return "processed"


def has_complete_artist_output(sources: list[Path], destination_dir: Path) -> bool:
    """Check whether destination already contains outputs for all source filenames."""
    if not destination_dir.exists():
        return False
    for source in sources:
        if not (destination_dir / source.name).exists():
            return False
    return True


def process_artist_method(
    method: str,
    target: ArtistTarget,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, int | str]:
    """Process one artist with one method and return summary counts."""
    sources = image_files(target.source_dir)
    destination_dir = target.output_root / method / target.split / target.style / target.artist
    processed = 0
    skipped = 0
    failed = 0

    if mode == "external_batch_artist":
        if has_complete_artist_output(sources=sources, destination_dir=destination_dir) and not overwrite:
            return {
                "status": "complete",
                "mode": mode,
                "source_count": len(sources),
                "processed_count": 0,
                "skipped_count": len(sources),
                "failed_count": 0,
            }
        try:
            run_external_method_batch(
                method=method,
                input_dir=target.source_dir,
                output_dir=destination_dir,
                dry_run=dry_run,
            )
        except (OSError, subprocess.CalledProcessError, FileNotFoundError) as error:
            print(f"  failed batch: {method}/{target.split}/{target.style}/{target.artist}: {error}")
            return {
                "status": "incomplete",
                "mode": mode,
                "source_count": len(sources),
                "processed_count": 0,
                "skipped_count": 0,
                "failed_count": len(sources),
            }

        if dry_run:
            return {
                "status": "complete",
                "mode": mode,
                "source_count": len(sources),
                "processed_count": len(sources),
                "skipped_count": 0,
                "failed_count": 0,
            }

        matched = sum(1 for source in sources if (destination_dir / source.name).exists())
        failed = len(sources) - matched
        return {
            "status": "complete" if failed == 0 else "incomplete",
            "mode": mode,
            "source_count": len(sources),
            "processed_count": matched,
            "skipped_count": 0,
            "failed_count": failed,
        }

    for source_image in sources:
        output_image = destination_dir / source_image.name
        try:
            status = apply_one_image(
                method=method,
                source_image=source_image,
                output_image=output_image,
                mode=mode,
                overwrite=overwrite,
                dry_run=dry_run,
            )
            if status == "processed":
                processed += 1
            else:
                skipped += 1
        except (OSError, subprocess.CalledProcessError, FileNotFoundError) as error:
            failed += 1
            print(f"  failed: {method}/{target.split}/{target.style}/{target.artist}/{source_image.name}: {error}")

    final_status = "complete" if failed == 0 else "incomplete"
    return {
        "status": final_status,
        "mode": mode,
        "source_count": len(sources),
        "processed_count": processed,
        "skipped_count": skipped,
        "failed_count": failed,
    }


def main() -> int:
    """Run Step 2 protection application and update progress tracker bookmarks."""
    args = parse_args()
    requested_methods = args.method
    if not requested_methods and args.mode == "external_batch_artist":
        requested_methods = ["mist"]
    methods = get_method_list(requested_methods)
    split_filter = sorted(dict.fromkeys(args.split))
    style_filter = set(args.style)
    artist_filter = set(args.artist)

    targets = discover_targets(
        data_root=args.data_root,
        splits=split_filter,
        style_filter=style_filter,
        artist_filter=artist_filter,
    )
    if args.limit_artists > 0:
        targets = targets[: args.limit_artists]

    tracker = ProgressTracker(
        tracker_path=args.data_root / "progress" / "progress_tracker.json",
        data_root=args.data_root,
        min_samples_per_artist=args.min_samples_per_artist,
    )
    tracker.load()

    # Remove corrupt/truncated images before any method touches them.
    total_removed: list[Path] = []
    for target in targets:
        total_removed.extend(remove_corrupt_images(target.source_dir))
    if total_removed:
        print(f"Removed {len(total_removed)} corrupt image(s) from source directories.")

    print(
        f"Applying protections for {len(targets)} artist targets across {len(methods)} method(s). "
        f"mode={args.mode}, overwrite={args.overwrite}, dry_run={args.dry_run}"
    )
    if args.mode == "placeholder_copy":
        print("WARNING: placeholder_copy mode does not apply real perturbations.")

    for method in methods:
        print(f"\nMethod: {method}")
        runner_available = True
        setup_error = ""
        runner_path = get_external_runner(method)
        if args.mode in {"external", "external_batch_artist"}:
            runner_available, setup_error = check_external_runner(method=method, runner_path=runner_path)
            if not runner_available:
                print(f"  external runner not ready: {setup_error}")

        for target in targets:
            bookmark_key = f"{method}/{target.split}/{target.style}/{target.artist}"
            print(f"  target: {target.split}/{target.style}/{target.artist}")
            if not runner_available:
                result = {
                    "status": "failed_setup",
                    "mode": args.mode,
                    "source_count": len(image_files(target.source_dir)),
                    "processed_count": 0,
                    "skipped_count": 0,
                    "failed_count": 0,
                    "error": setup_error,
                }
            else:
                result = process_artist_method(
                    method=method,
                    target=target,
                    mode=args.mode,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            if not args.dry_run:
                tracker.set_bookmark(
                    step_name="step_2_apply_protections",
                    bookmark_key=bookmark_key,
                    payload={
                        "method": method,
                        "split": target.split,
                        "style": target.style,
                        "artist": target.artist,
                        **result,
                    },
                )

    if args.dry_run:
        print("\nDry-run complete. Tracker bookmarks were not updated.")
        return 0

    tracker.refresh_snapshot()
    print(f"\nUpdated tracker: {args.data_root / 'progress' / 'progress_tracker.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
