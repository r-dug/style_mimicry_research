#!/usr/bin/env python3
"""Batch-friendly wrapper for mist-v2 protection generation."""

from __future__ import annotations

import argparse, re, shutil, subprocess, tempfile, sys
from pathlib import Path

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MIMICRY_DIR = _PROJECT_ROOT / "mimicry"
if str(_MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(_MIMICRY_DIR))

from common.constants import IMAGE_EXTENSIONS  # noqa: E402
from common.model_resolution import (  # noqa: E402
    add_model_args,
    ensure_runtime_dependencies,
    resolve_model_source,
)

IMAGE_EXTENSIONS_SET = set(IMAGE_EXTENSIONS)
MIST_OUTPUT_PATTERN = re.compile(r"(?P<epoch>\d+)_noise_(?P<name>.+)\.(?P<ext>png|jpg|jpeg)$", re.IGNORECASE)
LOG_PREFIX = "mist"


def parse_args() -> argparse.Namespace:
    """Parse wrapper arguments for file mode and artist-folder mode."""
    parser = argparse.ArgumentParser(description="Run mist-v2 on one image or one artist folder.")
    parser.add_argument("--input", type=Path, help="Single input image path.")
    parser.add_argument("--output", type=Path, help="Single output image path.")
    parser.add_argument("--input-dir", type=Path, help="Input artist folder path.")
    parser.add_argument("--output-dir", type=Path, help="Output artist folder path.")
    parser.add_argument(
        "--mist-root",
        type=Path,
        default=Path("mimicry/protections/tools/mist"),
        help="Root path of mist-v2 checkout.",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu", help="Execution device for mist-v2.")
    parser.add_argument("--mode", choices=("lunet", "fused", "anti-db"), default="lunet", help="Mist attack mode.")

    # Shared model args
    add_model_args(parser)
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
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--max-f-train-steps", type=int, default=2)
    parser.add_argument("--max-adv-train-steps", type=int, default=5)
    parser.add_argument("--checkpointing-iterations", type=int, default=1)
    parser.add_argument("--prior-loss-weight", type=float, default=0.1)
    parser.add_argument("--pgd-alpha", type=float, default=0.005)
    parser.add_argument("--pgd-eps", type=float, default=0.04)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize perturbed outputs back to original resolution.",
    )
    parser.add_argument(
        "--low-vram-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable low-VRAM mode in mist-v2.",
    )
    parser.add_argument(
        "--class-data-dir",
        type=Path,
        default=None,
        help="Persistent directory for prior-preservation class images. "
        "Reused across runs so class images are only generated once.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    """List images in a directory (non-recursive), sorted by filename."""
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS_SET)


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


def run_mist(
    args: argparse.Namespace,
    resolved_model_path: str,
    instance_dir: Path,
    output_dir: Path,
    class_dir: Path,
) -> None:
    """Run mist-v2 once for the given input folder."""
    # Resolve all paths to absolute so they remain valid when the subprocess
    # runs with cwd=args.mist_root.
    abs_instance_dir = str(instance_dir.resolve())
    abs_output_dir = str(output_dir.resolve())
    abs_class_dir = str(class_dir.resolve())

    command = [
        "python3",
        "-m",
        "accelerate.commands.launch",
        "attacks/mist.py",
        "--mode",
        args.mode,
        "--pretrained_model_name_or_path",
        resolved_model_path,
        "--instance_data_dir",
        abs_instance_dir,
        "--output_dir",
        abs_output_dir,
        "--class_data_dir",
        abs_class_dir,
        "--instance_prompt",
        args.instance_prompt,
        "--class_prompt",
        args.class_prompt,
        "--mixed_precision",
        args.mixed_precision,
        "--max_train_steps",
        str(args.max_train_steps),
        "--checkpointing_iterations",
        str(args.checkpointing_iterations),
        "--prior_loss_weight",
        str(args.prior_loss_weight),
        "--pgd_alpha",
        str(args.pgd_alpha),
        "--pgd_eps",
        str(args.pgd_eps),
        "--max_adv_train_steps",
        str(args.max_adv_train_steps),
        "--max_f_train_steps",
        str(args.max_f_train_steps),
        "--resolution",
        str(args.resolution),
    ]
    if args.device == "gpu":
        command.append("--cuda")
        if args.low_vram_mode:
            command.append("--low_vram_mode")
    if args.resize:
        command.append("--resize")

    subprocess.run(command, cwd=args.mist_root, check=True)


def save_with_source_suffix(source_noise: Path, destination: Path) -> None:
    """Save generated perturbation using destination filename extension."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_noise) as image:
        ext = destination.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
        elif ext == ".png":
            image.save(destination, format="PNG")
        else:
            image.convert("RGB").save(destination)


def collect_best_outputs(raw_output_dir: Path) -> dict[str, Path]:
    """Collect latest epoch mist outputs by source stem name."""
    best_by_stem: dict[str, tuple[int, Path]] = {}
    for path in list_images(raw_output_dir):
        match = MIST_OUTPUT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        stem = Path(match.group("name")).stem
        current = best_by_stem.get(stem)
        if current is None or epoch > current[0]:
            best_by_stem[stem] = (epoch, path)
    return {stem: data[1] for stem, data in best_by_stem.items()}


def normalize_mist_outputs(source_dir: Path, raw_output_dir: Path, final_output_dir: Path) -> tuple[int, int]:
    """Map mist output names back to original filenames."""
    source_images = list_images(source_dir)
    generated = collect_best_outputs(raw_output_dir=raw_output_dir)
    success = 0
    miss = 0

    for source in source_images:
        noise_path = generated.get(source.stem)
        if noise_path is None:
            miss += 1
            continue
        save_with_source_suffix(source_noise=noise_path, destination=final_output_dir / source.name)
        success += 1
    return success, miss


def run_dir_mode(
    args: argparse.Namespace,
    resolved_model_path: str,
    input_dir: Path,
    output_dir: Path,
) -> int:
    """Run mist-v2 once for one artist folder."""
    source_images = list_images(input_dir)
    if not source_images:
        print(f"[mist] no images found in {input_dir}")
        return 2

    # Use a persistent class data directory so expensive class-image
    # generation survives across retries.  MIST itself skips generation
    # when the directory already contains enough images.
    if args.class_data_dir is not None:
        class_dir = args.class_data_dir.resolve()
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"[mist] using persistent class data dir: {class_dir}")
    else:
        class_dir = None  # will be created inside temp dir below

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mist_batch_") as temp_dir:
        temp_root = Path(temp_dir)
        raw_output_dir = temp_root / "mist_raw_output"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        if class_dir is None:
            class_dir = temp_root / "class_data"
            class_dir.mkdir(parents=True, exist_ok=True)
        run_mist(
            args=args,
            resolved_model_path=resolved_model_path,
            instance_dir=input_dir,
            output_dir=raw_output_dir,
            class_dir=class_dir,
        )
        success, miss = normalize_mist_outputs(
            source_dir=input_dir,
            raw_output_dir=raw_output_dir,
            final_output_dir=output_dir,
        )

    print(f"[mist] processed folder {input_dir}: success={success}, missing={miss}")
    return 0 if miss == 0 else 1


def run_file_mode(
    args: argparse.Namespace,
    resolved_model_path: str,
    input_path: Path,
    output_path: Path,
) -> int:
    """Run mist-v2 for one image by wrapping into a temporary folder job."""
    with tempfile.TemporaryDirectory(prefix="mist_single_") as temp_dir:
        temp_root = Path(temp_dir)
        single_input_dir = temp_root / "input"
        single_output_dir = temp_root / "output"
        single_input_dir.mkdir(parents=True, exist_ok=True)
        single_output_dir.mkdir(parents=True, exist_ok=True)
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
        produced = single_output_dir / input_path.name
        if not produced.exists():
            print(f"[mist] expected output missing for {input_path.name}")
            return 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, output_path)
    print(f"[mist] wrote {output_path}")
    return 0


def main() -> int:
    """Entry point for Mist batch wrapper."""
    args = parse_args()
    if not args.mist_root.exists():
        print(f"[mist] mist root not found: {args.mist_root}")
        return 2
    deps_ok, missing = ensure_runtime_dependencies()
    if not deps_ok:
        print(f"[mist] missing python dependencies: {', '.join(missing)}")
        print("[mist] install with: python3 -m pip install -r mimicry/protections/tools/mist/requirements.txt")
        return 2
    try:
        mode, input_target, output_target = validate_mode_args(args=args)
    except ValueError as error:
        print(f"[mist] Mode Arg validation failure.\n\tOccurred at {sys._getframe().f_code.co_filename}:{sys._getframe().f_lineno}\n\tERROR: {error}")
        return 2
    try:
        resolved_model_path = resolve_model_source(args=args, log_prefix=LOG_PREFIX)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"[mist] Could not resolve model path.\n\tOccurred at {sys._getframe().f_code.co_filename}:{sys._getframe().f_lineno}\n\tERROR:{error}")
        return 2

    if mode == "dir":
        if not input_target.exists():
            print(f"[mist] input dir does not exist: {input_target}")
            return 2
        return run_dir_mode(
            args=args,
            resolved_model_path=resolved_model_path,
            input_dir=input_target,
            output_dir=output_target,
        )

    if not input_target.exists():
        print(f"[mist] input file does not exist: {input_target}")
        return 2
    return run_file_mode(
        args=args,
        resolved_model_path=resolved_model_path,
        input_path=input_target,
        output_path=output_target,
    )


if __name__ == "__main__":
    raise SystemExit(main())
