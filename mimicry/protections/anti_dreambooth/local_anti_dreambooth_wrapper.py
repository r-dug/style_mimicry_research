#!/usr/bin/env python3
"""Batch-friendly wrapper for Anti-DreamBooth (ASPL) protection generation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MIMICRY_DIR = _PROJECT_ROOT / "mimicry"
if str(_MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(_MIMICRY_DIR))

from common.constants import IMAGE_EXTENSIONS, DEFAULT_CLASS_DATA_DIR  # noqa: E402
from common.model_resolution import (  # noqa: E402
    add_model_args,
    ensure_runtime_dependencies,
    resolve_model_source,
)
from common.noise_ckpt_output import normalize_noise_ckpt_outputs  # noqa: E402

IMAGE_EXTENSIONS_SET = set(IMAGE_EXTENSIONS)
LOG_PREFIX = "anti_dreambooth"


def parse_args() -> argparse.Namespace:
    """Parse wrapper arguments for file mode and artist-folder mode."""
    parser = argparse.ArgumentParser(
        description="Run Anti-DreamBooth ASPL on one image or one artist folder.",
    )
    parser.add_argument("--input", type=Path, help="Single input image path.")
    parser.add_argument("--output", type=Path, help="Single output image path.")
    parser.add_argument("--input-dir", type=Path, help="Input artist folder path.")
    parser.add_argument("--output-dir", type=Path, help="Output artist folder path.")
    parser.add_argument(
        "--tool-root",
        type=Path,
        default=Path("mimicry/protections/tools/anti_dreambooth"),
        help="Root path of Anti-DreamBooth checkout.",
    )

    # Shared model args
    add_model_args(parser)

    # Anti-DreamBooth specific args
    parser.add_argument(
        "--instance-prompt",
        default="a painting, high quality, masterpiece",
        help="Prompt describing the protected images.",
    )
    parser.add_argument(
        "--class-prompt",
        default="a painting, high quality, masterpiece",
        help="Class prompt used for prior-preservation data.",
    )
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--max-train-steps", type=int, default=20)
    parser.add_argument("--max-f-train-steps", type=int, default=10)
    parser.add_argument("--max-adv-train-steps", type=int, default=10)
    parser.add_argument("--checkpointing-iterations", type=int, default=5)
    parser.add_argument("--pgd-alpha", type=float, default=1.0 / 255)
    parser.add_argument("--pgd-eps", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--train-batch-size", type=int, default=4)

    # Prior preservation
    parser.add_argument("--with-prior-preservation", action="store_true", default=False)
    parser.add_argument(
        "--class-data-dir",
        type=Path,
        default=DEFAULT_CLASS_DATA_DIR,
        help="Persistent directory for prior-preservation class images.",
    )
    parser.add_argument("--num-class-images", type=int, default=100)
    parser.add_argument("--prior-loss-weight", type=float, default=1.0)

    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    """List images in a directory (non-recursive), sorted by filename."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS_SET
    )


def validate_mode_args(args: argparse.Namespace) -> tuple[str, Path, Path]:
    """Validate file mode vs folder mode args and return normalized paths."""
    file_mode = args.input is not None or args.output is not None
    dir_mode = args.input_dir is not None or args.output_dir is not None
    if file_mode and dir_mode:
        raise ValueError("Use either --input/--output or --input-dir/--output-dir, not both.")
    if not file_mode and not dir_mode:
        raise ValueError("Provide either --input/--output or --input-dir/--output-dir.")
    if file_mode:
        if args.input is None or args.output is None:
            raise ValueError("Both --input and --output are required in file mode.")
        return "file", args.input, args.output
    if args.input_dir is None or args.output_dir is None:
        raise ValueError("Both --input-dir and --output-dir are required in folder mode.")
    return "dir", args.input_dir, args.output_dir


def run_anti_dreambooth(
    args: argparse.Namespace,
    resolved_model_path: str,
    instance_dir: Path,
    output_dir: Path,
) -> None:
    """Run Anti-DreamBooth ASPL once for the given input folder."""
    abs_instance = str(instance_dir.resolve())
    abs_output = str(output_dir.resolve())

    command = [
        "python3",
        "-m",
        "accelerate.commands.launch",
        "attacks/aspl.py",
        "--pretrained_model_name_or_path",
        resolved_model_path,
        "--instance_data_dir_for_train",
        abs_instance,
        "--instance_data_dir_for_adversarial",
        abs_instance,
        "--output_dir",
        abs_output,
        "--instance_prompt",
        args.instance_prompt,
        "--class_prompt",
        args.class_prompt,
        "--mixed_precision",
        args.mixed_precision,
        "--max_train_steps",
        str(args.max_train_steps),
        "--max_f_train_steps",
        str(args.max_f_train_steps),
        "--max_adv_train_steps",
        str(args.max_adv_train_steps),
        "--checkpointing_iterations",
        str(args.checkpointing_iterations),
        "--pgd_alpha",
        str(args.pgd_alpha),
        "--pgd_eps",
        str(args.pgd_eps),
        "--resolution",
        str(args.resolution),
        "--learning_rate",
        str(args.learning_rate),
        "--train_batch_size",
        str(args.train_batch_size),
    ]

    if args.with_prior_preservation:
        command.append("--with_prior_preservation")
        if args.class_data_dir is not None:
            command.extend(["--class_data_dir", str(args.class_data_dir.resolve())])
        command.extend(["--num_class_images", str(args.num_class_images)])
        command.extend(["--prior_loss_weight", str(args.prior_loss_weight)])

    subprocess.run(command, cwd=args.tool_root, check=True)


def run_dir_mode(
    args: argparse.Namespace,
    resolved_model_path: str,
    input_dir: Path,
    output_dir: Path,
) -> int:
    """Run Anti-DreamBooth for one artist folder."""
    source_images = list_images(input_dir)
    if not source_images:
        print(f"[{LOG_PREFIX}] no images found in {input_dir}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="antidb_batch_") as temp_dir:
        raw_output_dir = Path(temp_dir) / "raw_output"
        raw_output_dir.mkdir(parents=True, exist_ok=True)

        run_anti_dreambooth(
            args=args,
            resolved_model_path=resolved_model_path,
            instance_dir=input_dir,
            output_dir=raw_output_dir,
        )

        success, miss = normalize_noise_ckpt_outputs(
            source_dir=input_dir,
            raw_output_dir=raw_output_dir,
            final_output_dir=output_dir,
            log_prefix=LOG_PREFIX,
        )

    print(f"[{LOG_PREFIX}] processed {input_dir}: success={success}, missing={miss}")
    return 0 if miss == 0 else 1


def run_file_mode(
    args: argparse.Namespace,
    resolved_model_path: str,
    input_path: Path,
    output_path: Path,
) -> int:
    """Run Anti-DreamBooth for one image by wrapping into a temporary folder job."""
    with tempfile.TemporaryDirectory(prefix="antidb_single_") as temp_dir:
        temp_root = Path(temp_dir)
        single_input_dir = temp_root / "input"
        single_output_dir = temp_root / "output"
        single_input_dir.mkdir()
        single_output_dir.mkdir()
        staged_input = single_input_dir / input_path.name
        shutil.copy2(input_path, staged_input)
        code = run_dir_mode(
            args=args,
            resolved_model_path=resolved_model_path,
            input_dir=single_input_dir,
            output_dir=single_output_dir,
        )
        if code != 0:
            return code
        result = single_output_dir / input_path.name
        if not result.exists():
            print(f"[{LOG_PREFIX}] expected output not found: {result}")
            return 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, output_path)
    return 0


def main() -> int:
    """Entry point for Anti-DreamBooth batch wrapper."""
    args = parse_args()
    if not args.tool_root.exists():
        print(f"[{LOG_PREFIX}] tool root not found: {args.tool_root}")
        return 2

    deps_ok, missing = ensure_runtime_dependencies()
    if not deps_ok:
        print(f"[{LOG_PREFIX}] missing python dependencies: {', '.join(missing)}")
        return 2

    try:
        mode, input_target, output_target = validate_mode_args(args=args)
    except ValueError as error:
        print(f"[{LOG_PREFIX}] {error}")
        return 2

    try:
        resolved_model_path = resolve_model_source(args=args, log_prefix=LOG_PREFIX)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"[{LOG_PREFIX}] could not resolve model path: {error}")
        return 2

    if mode == "dir":
        if not input_target.exists():
            print(f"[{LOG_PREFIX}] input dir does not exist: {input_target}")
            return 2
        return run_dir_mode(
            args=args,
            resolved_model_path=resolved_model_path,
            input_dir=input_target,
            output_dir=output_target,
        )

    if not input_target.exists():
        print(f"[{LOG_PREFIX}] input file does not exist: {input_target}")
        return 2
    return run_file_mode(
        args=args,
        resolved_model_path=resolved_model_path,
        input_path=input_target,
        output_path=output_target,
    )


if __name__ == "__main__":
    raise SystemExit(main())
