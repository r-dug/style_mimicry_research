"""Experiment configuration getters for pipeline steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from common.constants import PROTECTED_METHODS, ROBUST_METHODS


AspectMode = Literal["center_crop", "pad"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Image preprocessing settings shared across training runs."""

    width: int = 1024
    height: int = 1024
    aspect_mode: AspectMode = "center_crop"


@dataclass(frozen=True)
class TrainingStackConfig:
    """Training-stack defaults for SD3.5 style mimicry fine-tuning."""

    sd35_variant: str = "large"
    stacks: tuple[str, str] = ("simpletuner", "hf_diffusers_dreambooth_lora_sd3")
    hf_push_to_hub: bool = True
    hf_private_repo: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration payload used across the project."""

    data_root: str = "data/style_mimicry"
    min_samples_per_artist: int = 20
    preprocess: PreprocessConfig = PreprocessConfig()
    training: TrainingStackConfig = TrainingStackConfig()
    protection_methods: tuple[str, ...] = PROTECTED_METHODS
    countermeasure_methods: tuple[str, ...] = ROBUST_METHODS


def get_pipeline_config() -> PipelineConfig:
    """Return immutable defaults for the full experiment pipeline."""
    return PipelineConfig()


def get_pipeline_config_dict() -> dict[str, object]:
    """Return the full pipeline config as a plain dictionary."""
    return asdict(get_pipeline_config())


def get_pilot_artists_by_style(data_root: Path, split: str = "historical") -> dict[str, str]:
    """Select one artist per style for trial runs."""
    base_dir = data_root / "original_art" / split
    pilots: dict[str, str] = {}
    if not base_dir.exists():
        return pilots

    for style_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        artist_dirs = sorted(path for path in style_dir.iterdir() if path.is_dir())
        if not artist_dirs:
            continue
        pilots[style_dir.name] = artist_dirs[0].name
    return pilots

