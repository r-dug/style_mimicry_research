"""Canonical constants for paths and method naming."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT_DEFAULT: Path = Path("data/style_mimicry")
DEFAULT_SD_MODEL: str = "runwayml/stable-diffusion-v1-5"
DEFAULT_MODEL_CACHE_DIR: Path = Path("data/style_mimicry/models")
DEFAULT_CLASS_DATA_DIR: Path = Path("data/style_mimicry/class_data")
PROTECTED_METHODS: tuple[str, ...] = (
    "glaze",
    "mist",
    "anti_dreambooth",
    "style_guard",
)
ROBUST_METHODS: tuple[str, ...] = (
    "diff_pure",
    "gaussian_noise",
    "noisy_upscaling",
    "impress",
)
DEFAULT_SD35_MODEL: str = "stabilityai/stable-diffusion-3.5-large"
TUNING_STACKS: tuple[str, ...] = ("hf_diffusers", "simpletuner")
DEFAULT_LORA_OUTPUT_DIR: Path = Path("data/style_mimicry/models/lora")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
PIPELINE_STEPS: tuple[str, ...] = (
    "step_1_collect_original_art",
    "step_2_apply_protections",
    "step_3_fine_tune_sd35",
    "step_4_generate_mimic_art",
    "step_5_algorithmic_benchmark",
    "step_6_human_survey",
    "step_7_final_analysis",
)

