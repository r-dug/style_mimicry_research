#!/usr/bin/env python3
"""Wrapper for DreamBooth LoRA fine-tuning on SD3.5 Large via SimpleTuner.

Generates per-artist config.json and multidatabackend.json in the output directory,
then invokes SimpleTuner via the CONFIG_PATH environment variable.

Install SimpleTuner:
    pip install 'simpletuner[cuda]'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MIMICRY_DIR = _PROJECT_ROOT / "mimicry"
if str(_MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(_MIMICRY_DIR))

from common.constants import DEFAULT_MODEL_CACHE_DIR, DEFAULT_SD35_MODEL, IMAGE_EXTENSIONS  # noqa: E402

IMAGE_EXTENSIONS_SET = set(IMAGE_EXTENSIONS)
LOG_PREFIX = "simpletuner_lora_sd3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DreamBooth LoRA fine-tuning on SD3.5 Large via SimpleTuner.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Artist image directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="LoRA output directory.")
    parser.add_argument("--pretrained-model-name-or-path", default=DEFAULT_SD35_MODEL)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR,
                        help="HF hub cache dir (set as HF_HUB_CACHE for SimpleTuner).")

    # Instance prompt
    parser.add_argument("--instance-prompt", default="a painting in the style of sks")

    # Training hyperparameters — single source of truth for this stack
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="polynomial",
                        choices=("constant", "cosine", "polynomial", "linear"))
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha scaling. Defaults to rank.")
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--snr-gamma", type=float, default=5.0,
                        help="Min-SNR loss weighting gamma (SimpleTuner feature).")
    parser.add_argument("--caption-dropout-probability", type=float, default=0.1)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=50,
                        help="Dataset repeat multiplier (inflates small dataset for sampling).")

    # Validation
    parser.add_argument("--validation-prompt", default=None,
                        help="Validation image prompt. Defaults to instance prompt.")
    parser.add_argument("--validation-steps", type=int, default=250)
    parser.add_argument("--checkpoint-steps", type=int, default=500)

    # HuggingFace Hub
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)

    # Naming metadata (from orchestrator, used for hub ID only)
    parser.add_argument("--protection", default="none")
    parser.add_argument("--countermeasure", default="none")

    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS_SET
    )


def build_hub_model_id(args: argparse.Namespace) -> str:
    artist = args.input_dir.resolve().name
    return f"sd35-simpletuner-lora-{args.protection}-{args.countermeasure}-{artist}"


def write_configs(args: argparse.Namespace) -> Path:
    """Write config.json and multidatabackend.json to output_dir. Returns config path."""
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.rank
    validation_prompt = args.validation_prompt or args.instance_prompt
    hub_model_id = args.hub_model_id or build_hub_model_id(args)

    multidatabackend_path = output_dir / "multidatabackend.json"
    config_path = output_dir / "config.json"

    multidatabackend = [
        {
            "id": "artist_training",
            "type": "local",
            "instance_data_dir": str(args.input_dir.resolve()),
            "dataset_type": "image",
            "resolution_type": "pixel",
            "minimum_image_size": 512,
            "resolution": args.resolution,
            "crop": True,
            "crop_aspect": "square",
            "caption_strategy": "instanceprompt",
            "instance_prompt": args.instance_prompt,
            "repeats": args.repeats,
            "cache_dir_vae": str(output_dir / "vae_cache"),
        }
    ]

    config = {
        "model_family": "stable-diffusion-3",
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "model_type": "lora",
        "lora_type": "standard",
        "lora_rank": args.rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_steps": args.lr_warmup_steps,
        "optimizer": "adamw_bf16",
        "max_train_steps": args.max_train_steps,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "mixed_precision": args.mixed_precision,
        "resolution": args.resolution,
        "caption_dropout_probability": args.caption_dropout_probability,
        "snr_gamma": args.snr_gamma,
        "seed": args.seed,
        "validation_step_interval": args.validation_steps,
        "num_validation_images": 2,
        "validation_prompt": validation_prompt,
        "validation_guidance_scale": 7.0,
        "validation_inference_steps": 28,
        "checkpoint_step_interval": args.checkpoint_steps,
        "max_checkpoints": 3,
        "use_ema": False,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": hub_model_id,
        "hub_private_repo": args.private,
        "report_to": "tensorboard",
        "data_backend_config": str(multidatabackend_path),
        "output_dir": str(output_dir),
    }

    with open(multidatabackend_path, "w") as f:
        json.dump(multidatabackend, f, indent=2)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"[{LOG_PREFIX}] input dir not found: {args.input_dir}")
        return 2

    images = list_images(args.input_dir)
    if not images:
        print(f"[{LOG_PREFIX}] no images found in {args.input_dir}")
        return 2

    config_path = write_configs(args)

    print(f"[{LOG_PREFIX}] training on {len(images)} images from {args.input_dir}")
    print(f"[{LOG_PREFIX}] output: {args.output_dir}")
    print(f"[{LOG_PREFIX}] config: {config_path}")

    env = os.environ.copy()
    env["HF_HUB_CACHE"] = str(args.model_cache_dir.expanduser().resolve())
    env["CONFIG_PATH"] = str(config_path)

    try:
        subprocess.run(["simpletuner"], env=env, check=True)
    except FileNotFoundError:
        print(f"[{LOG_PREFIX}] 'simpletuner' not found. Install with: pip install 'simpletuner[cuda]'")
        return 2
    except subprocess.CalledProcessError as error:
        print(f"[{LOG_PREFIX}] training failed with exit code {error.returncode}")
        return error.returncode

    lora_weights = args.output_dir / "pytorch_lora_weights.safetensors"
    if not lora_weights.exists():
        print(f"[{LOG_PREFIX}] WARNING: expected LoRA weights not found at {lora_weights}")
        return 1

    print(f"[{LOG_PREFIX}] training complete. LoRA weights: {lora_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
