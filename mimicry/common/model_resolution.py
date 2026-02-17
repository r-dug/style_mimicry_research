"""Shared model resolution and HuggingFace download helpers."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from common.constants import DEFAULT_SD_MODEL, DEFAULT_MODEL_CACHE_DIR


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard model-related CLI arguments to *parser*."""
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default=DEFAULT_SD_MODEL,
        help="Model identifier/path for Stable Diffusion.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE_DIR,
        help="Local cache directory for Hugging Face model snapshots.",
    )
    parser.add_argument(
        "--auto-download-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-download model snapshot when not found locally/in cache.",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional Hugging Face token.",
    )
    parser.add_argument(
        "--hf-enable-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hf_transfer-accelerated downloads when available.",
    )
    parser.add_argument(
        "--hf-max-workers",
        type=int,
        default=16,
        help="Parallel download workers used by snapshot_download.",
    )


def ensure_runtime_dependencies() -> tuple[bool, list[str]]:
    """Check that core ML runtime dependencies are importable."""
    required_modules = ("accelerate", "torch", "diffusers", "transformers", "datasets")
    missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
    return len(missing) == 0, missing


def load_hf_token(args: argparse.Namespace) -> tuple[str | None, str]:
    """Resolve Hugging Face token from explicit CLI flag or environment."""
    if args.hf_token.strip():
        return args.hf_token.strip(), "cli"
    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(env_key, "").strip()
        if value:
            return value, f"env:{env_key}"
    return None, "default"


def configure_hf_download_env(args: argparse.Namespace) -> None:
    """Configure optional Hugging Face download acceleration environment flags."""
    if args.hf_enable_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def snapshot_download_safe(snapshot_download, **kwargs) -> str:
    """Call snapshot_download with compatibility fallback for older versions."""
    try:
        return snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("max_workers", None)
        return snapshot_download(**kwargs)


def get_model_candidates(model_spec: str) -> list[str]:
    """Return model id candidates for backwards-compatible SD1.5 aliases."""
    normalized = model_spec.strip()
    if normalized == "stable-diffusion/stable-diffusion-1-5":
        return [normalized, "runwayml/stable-diffusion-v1-5"]
    return [normalized]


def resolve_model_source(args: argparse.Namespace, log_prefix: str = "model") -> str:
    """Resolve model source to a local path or cached/downloaded HF snapshot.

    Returns an absolute path string.
    """
    model_spec = args.pretrained_model_name_or_path.strip()
    if not model_spec:
        raise ValueError("Empty --pretrained-model-name-or-path.")

    candidate = Path(model_spec).expanduser()
    if candidate.exists():
        resolved = str(candidate.resolve())
        print(f"[{log_prefix}] using local model path: {resolved}")
        return resolved

    looks_like_local_path = model_spec.startswith(("/", ".", "~")) or ("\\" in model_spec)
    if looks_like_local_path:
        raise FileNotFoundError(f"Local model path does not exist: {candidate}")

    if importlib.util.find_spec("huggingface_hub") is None:
        raise RuntimeError(
            "huggingface_hub is required to resolve non-local model identifiers. "
            "Install it or provide a local --pretrained-model-name-or-path."
        )

    from huggingface_hub import snapshot_download

    configure_hf_download_env(args=args)
    token, token_source = load_hf_token(args=args)
    if token_source == "cli" or token_source.startswith("env:"):
        print(f"[{log_prefix}] Hugging Face token source: {token_source}")
    else:
        print(f"[{log_prefix}] Hugging Face auth: default resolution (HF_TOKEN/env or cached login).")

    cache_dir = args.model_cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for candidate_model in get_model_candidates(model_spec=model_spec):
        try:
            cached = snapshot_download_safe(
                snapshot_download,
                repo_id=candidate_model,
                cache_dir=str(cache_dir),
                local_files_only=True,
                token=token,
                max_workers=max(1, args.hf_max_workers),
            )
            cached = str(Path(cached).resolve())
            print(f"[{log_prefix}] using cached model snapshot: {cached}")
            return cached
        except Exception as error:
            errors.append(f"{candidate_model} cache lookup: {type(error).__name__}: {error}")
            if not args.auto_download_model:
                continue
        if not args.auto_download_model:
            continue
        try:
            downloaded = snapshot_download_safe(
                snapshot_download,
                repo_id=candidate_model,
                cache_dir=str(cache_dir),
                local_files_only=False,
                token=token,
                max_workers=max(1, args.hf_max_workers),
            )
            downloaded = str(Path(downloaded).resolve())
            print(f"[{log_prefix}] downloaded model snapshot to cache: {downloaded}")
            return downloaded
        except Exception as error:
            errors.append(f"{candidate_model} download: {type(error).__name__}: {error}")

    if not args.auto_download_model:
        raise RuntimeError(
            f"Model '{model_spec}' not present in cache and --no-auto-download-model is set. "
            f"Tried candidates: {', '.join(get_model_candidates(model_spec))}"
        )
    raise RuntimeError(
        f"Failed to resolve/download model '{model_spec}'. "
        f"Tried candidates: {', '.join(get_model_candidates(model_spec))}. "
        f"Details: {' | '.join(errors)}"
    )
