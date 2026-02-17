"""Validate image files and remove corrupt/truncated entries."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from common.constants import IMAGE_EXTENSIONS


def validate_image(path: Path) -> bool:
    """Return True if *path* can be fully loaded as an RGB image."""
    try:
        with Image.open(path) as img:
            img.load()
            img.convert("RGB")
        return True
    except Exception:
        return False


def remove_corrupt_images(directory: Path, *, recursive: bool = False) -> list[Path]:
    """Delete corrupt or truncated images in *directory* and return removed paths.

    Only files whose suffix matches ``IMAGE_EXTENSIONS`` are tested.
    """
    removed: list[Path] = []
    pattern = directory.rglob("*") if recursive else directory.iterdir()
    for path in sorted(pattern):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not validate_image(path):
            print(f"[image_validation] removing corrupt image: {path}")
            path.unlink()
            removed.append(path)
    return removed
