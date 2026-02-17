"""Normalize noise-ckpt output format used by Anti-DreamBooth and StyleGuard.

Both tools save perturbed images to::

    {output_dir}/noise-ckpt/{iteration}/{iteration}_noise_{original_filename}

This module finds the highest iteration checkpoint and maps outputs back to
the original source filenames.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from common.constants import IMAGE_EXTENSIONS

IMAGE_EXTENSIONS_SET = set(IMAGE_EXTENSIONS)

NOISE_CKPT_PATTERN = re.compile(r"(?P<iteration>\d+)_noise_(?P<name>.+)$")


def save_with_source_suffix(source_noise: Path, destination: Path) -> None:
    """Save generated perturbation using *destination* filename extension."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_noise) as image:
        ext = destination.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
        elif ext == ".png":
            image.save(destination, format="PNG")
        else:
            image.convert("RGB").save(destination)


def find_highest_iteration_dir(raw_output_dir: Path) -> Path | None:
    """Return the noise-ckpt subdirectory with the highest iteration number."""
    noise_ckpt_dir = raw_output_dir / "noise-ckpt"
    if not noise_ckpt_dir.exists():
        return None
    iteration_dirs = [
        (int(d.name), d) for d in noise_ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()
    ]
    if not iteration_dirs:
        return None
    iteration_dirs.sort(key=lambda x: x[0], reverse=True)
    return iteration_dirs[0][1]


def collect_noise_ckpt_outputs(iteration_dir: Path) -> dict[str, Path]:
    """Map original filename stems to noise-ckpt output paths in *iteration_dir*."""
    results: dict[str, Path] = {}
    for path in sorted(iteration_dir.iterdir()):
        if not path.is_file():
            continue
        match = NOISE_CKPT_PATTERN.match(path.name)
        if match:
            original_name = match.group("name")
            stem = Path(original_name).stem
            results[stem] = path
    return results


def list_images(directory: Path) -> list[Path]:
    """List images in *directory* (non-recursive), sorted by filename."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS_SET
    )


def normalize_noise_ckpt_outputs(
    source_dir: Path,
    raw_output_dir: Path,
    final_output_dir: Path,
    log_prefix: str = "normalize",
) -> tuple[int, int]:
    """Map noise-ckpt output files back to original filenames in *final_output_dir*.

    Returns ``(success_count, miss_count)``.
    """
    source_images = list_images(source_dir)
    iteration_dir = find_highest_iteration_dir(raw_output_dir)
    if iteration_dir is None:
        print(f"[{log_prefix}] no noise-ckpt output directories found in {raw_output_dir}")
        return 0, len(source_images)

    generated = collect_noise_ckpt_outputs(iteration_dir)
    success = 0
    miss = 0

    for source in source_images:
        noise_path = generated.get(source.stem)
        if noise_path is None:
            print(f"[{log_prefix}] missing output for {source.name}")
            miss += 1
            continue
        save_with_source_suffix(source_noise=noise_path, destination=final_output_dir / source.name)
        success += 1
    return success, miss
