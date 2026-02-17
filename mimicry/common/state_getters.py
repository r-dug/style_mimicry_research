"""Filesystem state getters used for progress tracking and skip logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import IMAGE_EXTENSIONS, PROTECTED_METHODS, ROBUST_METHODS


def count_images(directory: Path) -> int:
    """Count image files recursively in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def get_original_art_state(data_root: Path, min_samples_per_artist: int = 20) -> dict[str, Any]:
    """Collect current original-art dataset coverage and threshold gaps."""
    base_dir = data_root / "original_art"
    state: dict[str, Any] = {
        "base_dir": str(base_dir),
        "min_samples_per_artist": min_samples_per_artist,
        "splits": {},
        "totals": {
            "styles": 0,
            "artists": 0,
            "images": 0,
            "artists_below_threshold": 0,
        },
        "artists_below_threshold": [],
    }

    for split_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()) if base_dir.exists() else []:
        split_name = split_dir.name
        split_styles: dict[str, Any] = {}
        split_stats = {"styles": 0, "artists": 0, "images": 0, "artists_below_threshold": 0}

        for style_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            style_name = style_dir.name
            style_artists: dict[str, int] = {}
            style_images = 0

            for artist_dir in sorted(path for path in style_dir.iterdir() if path.is_dir()):
                artist_name = artist_dir.name
                artist_count = count_images(artist_dir)
                style_artists[artist_name] = artist_count
                style_images += artist_count

                if artist_count < min_samples_per_artist:
                    state["artists_below_threshold"].append(
                        {
                            "split": split_name,
                            "style": style_name,
                            "artist": artist_name,
                            "count": artist_count,
                        }
                    )
                    split_stats["artists_below_threshold"] += 1

            split_styles[style_name] = {
                "artists": style_artists,
                "artist_count": len(style_artists),
                "image_count": style_images,
            }
            split_stats["styles"] += 1
            split_stats["artists"] += len(style_artists)
            split_stats["images"] += style_images

        state["splits"][split_name] = {
            "styles": split_styles,
            "style_count": split_stats["styles"],
            "artist_count": split_stats["artists"],
            "image_count": split_stats["images"],
            "artists_below_threshold": split_stats["artists_below_threshold"],
        }
        state["totals"]["styles"] += split_stats["styles"]
        state["totals"]["artists"] += split_stats["artists"]
        state["totals"]["images"] += split_stats["images"]
        state["totals"]["artists_below_threshold"] += split_stats["artists_below_threshold"]

    return state


def get_method_state(base_dir: Path, methods: tuple[str, ...]) -> dict[str, Any]:
    """Collect method-level counts from a two-level method/split data tree."""
    state: dict[str, Any] = {
        "base_dir": str(base_dir),
        "methods": {},
        "totals": {"methods": 0, "images": 0},
    }

    for method in methods:
        method_dir = base_dir / method
        split_counts: dict[str, int] = {}
        for split_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()) if method_dir.exists() else []:
            split_counts[split_dir.name] = count_images(split_dir)

        method_images = sum(split_counts.values())
        state["methods"][method] = {
            "exists": method_dir.exists(),
            "split_image_counts": split_counts,
            "image_count": method_images,
        }
        state["totals"]["methods"] += 1
        state["totals"]["images"] += method_images

    return state


def get_protected_art_state(data_root: Path) -> dict[str, Any]:
    """Collect current coverage for protected-art outputs."""
    return get_method_state(data_root / "protected_art", PROTECTED_METHODS)


def get_robust_samples_state(data_root: Path) -> dict[str, Any]:
    """Collect current coverage for robust sample outputs."""
    return get_method_state(data_root / "robust_samples", ROBUST_METHODS)


def get_mimic_art_state(data_root: Path) -> dict[str, Any]:
    """Collect current coverage for generated mimic-art samples."""
    mimic_dir = data_root / "mimic_art"
    return {
        "base_dir": str(mimic_dir),
        "image_count": count_images(mimic_dir),
    }

