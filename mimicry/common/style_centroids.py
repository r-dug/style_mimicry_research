"""Style centroid I/O and dissimilarity utilities for StyleGuard auto-target selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ArtistCentroid:
    """VAE latent-space statistics for one artist's image collection."""

    artist: str
    style: str
    category: str
    mean: list[float]
    var: list[float]
    image_count: int
    image_dir: str


def save_centroids(
    centroids: list[ArtistCentroid],
    path: Path,
    model: str,
    resolution: int,
) -> None:
    """Persist centroid data to a JSON file."""
    payload = {
        "model": model,
        "resolution": resolution,
        "centroids": [asdict(c) for c in centroids],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_centroids(path: Path) -> tuple[list[ArtistCentroid], dict]:
    """Load centroids from a JSON file.

    Returns ``(centroids, metadata)`` where metadata contains ``model`` and
    ``resolution`` keys.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    centroids = [ArtistCentroid(**entry) for entry in payload["centroids"]]
    metadata = {"model": payload["model"], "resolution": payload["resolution"]}
    return centroids, metadata


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def find_most_dissimilar(
    source: ArtistCentroid,
    candidates: list[ArtistCentroid],
) -> ArtistCentroid:
    """Return the candidate whose mean vector is most dissimilar to *source*.

    Excludes the source artist by name. Uses cosine distance on mean vectors.
    """
    best: ArtistCentroid | None = None
    best_dist = -1.0
    for candidate in candidates:
        if candidate.artist == source.artist:
            continue
        dist = _cosine_distance(source.mean, candidate.mean)
        if dist > best_dist:
            best_dist = dist
            best = candidate
    if best is None:
        raise ValueError(f"No valid target candidates for artist '{source.artist}'")
    return best
