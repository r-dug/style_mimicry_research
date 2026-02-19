#!/usr/bin/env python3
"""Wrapper for DreamBooth LoRA fine-tuning on SD3.5 Large via HuggingFace diffusers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MIMICRY_DIR = _PROJECT_ROOT / "mimicry"
if str(_MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(_MIMICRY_DIR))

from common.constants import DEFAULT_SD35_MODEL, IMAGE_EXTENSIONS  # noqa: E402
from common.model_resolution import (  # noqa: E402
    add_model_args,
    ensure_runtime_dependencies,
    resolve_model_source,
)

IMAGE_EXTENSIONS_SET = set(IMAGE_EXTENSIONS)
LOG_PREFIX = "hf_diffusers_lora_sd3"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_dreambooth_lora_sd3.py"


def parse_args() -> argparse.Namespace:
    """Parse wrapper CLI arguments."""
    parser = argparse.ArgumentParser(
        description="DreamBooth LoRA fine-tuning on SD3.5 Large.",
    )
    # Mode
    parser.add_argument("--input-dir", type=Path, required=True, help="Artist image directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="LoRA output directory.")

    # Shared model args (adds --pretrained-model-name-or-path, --model-cache-dir, etc.)
    add_model_args(parser)
    parser.set_defaults(pretrained_model_name_or_path=DEFAULT_SD35_MODEL)

    # Instance prompt
    parser.add_argument(
        "--instance-prompt",
        default="a painting in the style of sks",
        help="DreamBooth instance prompt with rare token identifier.",
    )

    # Training hyperparameters
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=4.25e-4)
    parser.add_argument("--lr-scheduler", default="cosine",
                        choices=("constant", "cosine", "linear", "polynomial"))
    parser.add_argument("--lr-warmup-steps", type=int, default=0,
                        help="LR warmup steps. 0 = auto (10%% of max steps).")
    parser.add_argument("--max-train-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=256, help="LoRA rank.")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-latents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache VAE latents to save VRAM during training.",
    )

    # HuggingFace Hub
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hub-model-id", default=None, help="HF Hub repo name (auto-generated if omitted).")
    parser.add_argument("--hub-token", default=None, help="HF token (falls back to HF_TOKEN env var).")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)

    # Naming metadata (forwarded from orchestrator, used for hub naming only)
    parser.add_argument("--protection", default="none", help="Protection method name (for naming).")
    parser.add_argument("--countermeasure", default="none", help="Countermeasure name (for naming).")

    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    """List images in a directory, sorted by filename."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS_SET
    )


def build_hub_model_id(args: argparse.Namespace) -> str:
    """Auto-generate a Hub model ID from the artist directory and variant."""
    artist = args.input_dir.resolve().name
    return f"sd35-lora-{args.protection}-{args.countermeasure}-{artist}"


def build_training_command(
    args: argparse.Namespace,
    resolved_model_path: str,
) -> list[str]:
    """Build the accelerate launch command for the vendored training script."""
    cmd = [
        "python3", "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        str(TRAIN_SCRIPT),
        "--pretrained_model_name_or_path", resolved_model_path,
        "--instance_data_dir", str(args.input_dir.resolve()),
        "--instance_prompt", args.instance_prompt,
        "--output_dir", str(args.output_dir.resolve()),
        "--resolution", str(args.resolution),
        "--train_batch_size", str(args.train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", args.lr_scheduler,
        "--max_train_steps", str(args.max_train_steps),
        "--rank", str(args.rank),
        "--mixed_precision", args.mixed_precision,
        "--seed", str(args.seed),
    ]

    # LR warmup: auto = 10% of max steps
    warmup = args.lr_warmup_steps if args.lr_warmup_steps > 0 else args.max_train_steps // 10
    cmd.extend(["--lr_warmup_steps", str(warmup)])

    # Note: snr_gamma is a SimpleTuner feature, not supported by the HF script

    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.center_crop:
        cmd.append("--center_crop")
    if args.cache_latents:
        cmd.append("--cache_latents")

    if args.push_to_hub:
        hub_model_id = args.hub_model_id or build_hub_model_id(args)
        cmd.append("--push_to_hub")
        cmd.extend(["--hub_model_id", hub_model_id])
        if args.hub_token:
            cmd.extend(["--hub_token", args.hub_token])
    return cmd


def main() -> int:
    """Entry point for DreamBooth LoRA SD3.5 wrapper."""
    args = parse_args()

    if not args.input_dir.exists():
        print(f"[{LOG_PREFIX}] input dir not found: {args.input_dir}")
        return 2

    images = list_images(args.input_dir)
    if not images:
        print(f"[{LOG_PREFIX}] no images found in {args.input_dir}")
        return 2

    deps_ok, missing = ensure_runtime_dependencies()
    if not deps_ok:
        print(f"[{LOG_PREFIX}] missing python dependencies: {', '.join(missing)}")
        return 2

    # Ensure model is downloaded/cached, but pass the original HF model ID
    # to the training script — it does its own model loading and needs the Hub ID
    # to properly resolve subfolder configs (text_encoder/config.json architectures).
    try:
        resolve_model_source(args=args, log_prefix=LOG_PREFIX)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"[{LOG_PREFIX}] could not resolve model: {error}")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = build_training_command(args, args.pretrained_model_name_or_path)

    print(f"[{LOG_PREFIX}] training on {len(images)} images from {args.input_dir}")
    print(f"[{LOG_PREFIX}] output: {args.output_dir}")
    print(f"[{LOG_PREFIX}] command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(f"[{LOG_PREFIX}] training failed with exit code {error.returncode}")
        return error.returncode

    # Verify output
    lora_weights = args.output_dir / "pytorch_lora_weights.safetensors"
    if not lora_weights.exists():
        print(f"[{LOG_PREFIX}] WARNING: expected LoRA weights not found at {lora_weights}")
        return 1

    print(f"[{LOG_PREFIX}] training complete. LoRA weights: {lora_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
