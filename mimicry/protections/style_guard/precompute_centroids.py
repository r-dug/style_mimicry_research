#!/usr/bin/env python3
"""Pre-compute VAE latent-space centroids for all artists.

Walks the original_art directory tree (category/style/artist/) and encodes
each artist's images through the Stable Diffusion VAE.  Saves per-artist
channel-wise mean and variance vectors to a JSON file that the StyleGuard
wrapper can use for automatic target selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MIMICRY_DIR = _PROJECT_ROOT / "mimicry"
if str(_MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(_MIMICRY_DIR))

from common.constants import DATA_ROOT_DEFAULT, IMAGE_EXTENSIONS  # noqa: E402
from common.model_resolution import add_model_args, resolve_model_source  # noqa: E402
from common.style_centroids import ArtistCentroid, save_centroids  # noqa: E402

IMAGE_EXTENSIONS_SET = set(IMAGE_EXTENSIONS)
LOG_PREFIX = "precompute_centroids"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute VAE latent centroids for all artists.",
    )
    parser.add_argument(
        "--art-root",
        type=Path,
        default=DATA_ROOT_DEFAULT / "original_art",
        help="Root directory containing category/style/artist folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT_DEFAULT / "style_centroids.json",
        help="Output path for the centroids JSON file.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Max images to encode at once (controls GPU memory).",
    )
    add_model_args(parser)
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    """List image files in *directory*, sorted by name."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS_SET
    )


def discover_artists(art_root: Path) -> list[tuple[str, str, str, Path]]:
    """Walk art_root and return (category, style, artist, path) tuples."""
    artists: list[tuple[str, str, str, Path]] = []
    for category_dir in sorted(art_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for style_dir in sorted(category_dir.iterdir()):
            if not style_dir.is_dir():
                continue
            for artist_dir in sorted(style_dir.iterdir()):
                if not artist_dir.is_dir():
                    continue
                if list_images(artist_dir):
                    artists.append((
                        category_dir.name,
                        style_dir.name,
                        artist_dir.name,
                        artist_dir,
                    ))
    return artists


def load_and_preprocess(path: Path, resolution: int) -> torch.Tensor:
    """Load, resize, center-crop, and normalize an image to [-1, 1].

    Replicates the torchvision transform chain used by styleguard.py
    (Resize -> CenterCrop -> ToTensor -> Normalize([0.5],[0.5])) using
    only PIL and torch, avoiding the torchvision import that can trigger
    libstdc++ issues via torch._dynamo/optree.
    """
    img = Image.open(path).convert("RGB")
    # Resize shortest side to resolution (bilinear), matching transforms.Resize
    w, h = img.size
    if h < w:
        new_h = resolution
        new_w = int(round(w * resolution / h))
    else:
        new_w = resolution
        new_h = int(round(h * resolution / w))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    # Center crop to resolution x resolution
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))
    # ToTensor equivalent: HWC uint8 [0,255] -> CHW float [0,1]
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    # Normalize([0.5],[0.5]) -> maps [0,1] to [-1,1]
    tensor = tensor * 2.0 - 1.0
    return tensor


@torch.no_grad()
def compute_centroid(
    image_paths: list[Path],
    vae,
    resolution: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[float], list[float]]:
    """Encode images through VAE and return (mean, var) channel-wise vectors."""
    all_means: list[torch.Tensor] = []
    all_vars: list[torch.Tensor] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = torch.stack([
            load_and_preprocess(p, resolution) for p in batch_paths
        ])
        latents = vae.encode(images.to(device, dtype=dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        # Channel-wise stats across batch and spatial dims — matches styleguard.py
        all_means.append(latents.mean(dim=[0, 2, 3]).cpu())
        all_vars.append(latents.var(dim=[0, 2, 3]).cpu())

    # Average the per-batch stats (weighted equally since we want the overall centroid)
    mean_vec = torch.stack(all_means).mean(dim=0)
    var_vec = torch.stack(all_vars).mean(dim=0)
    return mean_vec.tolist(), var_vec.tolist()


def main() -> int:
    args = parse_args()

    args.art_root = args.art_root.resolve()
    if not args.art_root.exists():
        print(f"[{LOG_PREFIX}] art root not found: {args.art_root}")
        return 2

    artists = discover_artists(args.art_root)
    if not artists:
        print(f"[{LOG_PREFIX}] no artist folders with images found under {args.art_root}")
        return 2
    print(f"[{LOG_PREFIX}] found {len(artists)} artists with images")

    try:
        resolved_model_path = resolve_model_source(args=args, log_prefix=LOG_PREFIX)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"[{LOG_PREFIX}] could not resolve model: {error}")
        return 2

    from diffusers import AutoencoderKL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[{LOG_PREFIX}] loading VAE from {resolved_model_path} on {device} ({dtype})")
    vae = AutoencoderKL.from_pretrained(resolved_model_path, subfolder="vae")
    vae.to(device, dtype=dtype)
    vae.eval()

    centroids: list[ArtistCentroid] = []

    for idx, (category, style, artist, artist_dir) in enumerate(artists, 1):
        images = list_images(artist_dir)
        print(f"[{LOG_PREFIX}] [{idx}/{len(artists)}] {category}/{style}/{artist} ({len(images)} images)")
        mean, var = compute_centroid(
            image_paths=images,
            vae=vae,
            resolution=args.resolution,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
        )
        rel_dir = str(artist_dir.relative_to(_PROJECT_ROOT))
        centroids.append(ArtistCentroid(
            artist=artist,
            style=style,
            category=category,
            mean=mean,
            var=var,
            image_count=len(images),
            image_dir=rel_dir,
        ))

    save_centroids(
        centroids=centroids,
        path=args.output,
        model=args.pretrained_model_name_or_path,
        resolution=args.resolution,
    )
    print(f"[{LOG_PREFIX}] saved {len(centroids)} centroids to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
