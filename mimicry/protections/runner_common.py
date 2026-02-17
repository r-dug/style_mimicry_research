"""Shared helpers for method-specific protection runners."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for invoking an external protection command."""

    command_template: str
    description: str = ""
    supports_file_mode: bool = False
    supports_dir_mode: bool = False


def load_runner_config(config_path: Path) -> RunnerConfig:
    """Load runner configuration from JSON."""
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    command_template = payload.get("command_template", "")
    if not isinstance(command_template, str) or not command_template.strip():
        raise ValueError("runner config must include non-empty 'command_template'.")
    supports_file_mode = "{input}" in command_template and "{output}" in command_template
    supports_dir_mode = "{input_dir}" in command_template and "{output_dir}" in command_template
    if not supports_file_mode and not supports_dir_mode:
        raise ValueError(
            "command_template must include either {input}/{output} or {input_dir}/{output_dir} placeholders."
        )

    description = payload.get("description", "")
    if description is None:
        description = ""
    return RunnerConfig(
        command_template=command_template,
        description=str(description),
        supports_file_mode=supports_file_mode,
        supports_dir_mode=supports_dir_mode,
    )


def build_parser(method: str, default_config_path: Path) -> argparse.ArgumentParser:
    """Build CLI parser used by method-specific runners."""
    parser = argparse.ArgumentParser(description=f"{method}: apply protection to one input image.")
    parser.add_argument("--input", type=Path, help="Input image path.")
    parser.add_argument("--output", type=Path, help="Output image path.")
    parser.add_argument("--input-dir", type=Path, help="Input image directory path.")
    parser.add_argument("--output-dir", type=Path, help="Output image directory path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help=f"Runner config JSON path (default: {default_config_path}).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate runner config and exit without processing an image.",
    )
    return parser


def run_method_runner(method: str, default_config_path: Path) -> int:
    """Run one method runner from config, or perform setup preflight."""
    parser = build_parser(method=method, default_config_path=default_config_path)
    args = parser.parse_args()

    if not args.config.exists():
        print(
            f"[{method}] missing runner config: {args.config}. "
            f"Create this file with a command_template before external mode."
        )
        return 2

    try:
        config = load_runner_config(args.config)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"[{method}] invalid runner config at {args.config}: {error}")
        return 2

    if args.check_only:
        print(f"[{method}] runner config validated: {args.config}")
        return 0

    file_mode = args.input is not None or args.output is not None
    dir_mode = args.input_dir is not None or args.output_dir is not None
    if file_mode and dir_mode:
        parser.error("Use either file mode (--input/--output) or dir mode (--input-dir/--output-dir), not both.")
        return 2
    if not file_mode and not dir_mode:
        parser.error("Provide either --input/--output or --input-dir/--output-dir.")
        return 2

    if file_mode:
        if args.input is None or args.output is None:
            parser.error("Both --input and --output are required for file mode.")
            return 2
        if not config.supports_file_mode:
            print(
                f"[{method}] runner config does not support file mode placeholders "
                f"({args.config})."
            )
            return 2
    if dir_mode:
        if args.input_dir is None or args.output_dir is None:
            parser.error("Both --input-dir and --output-dir are required for dir mode.")
            return 2
        if not config.supports_dir_mode:
            print(
                f"[{method}] runner config does not support dir mode placeholders "
                f"({args.config})."
            )
            return 2

    format_args = {
        "input": str(args.input) if args.input is not None else "",
        "output": str(args.output) if args.output is not None else "",
        "input_dir": str(args.input_dir) if args.input_dir is not None else "",
        "output_dir": str(args.output_dir) if args.output_dir is not None else "",
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    command_text = config.command_template.format(**format_args)
    command = shlex.split(command_text)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(f"[{method}] external command failed with exit code {error.returncode}: {command_text}")
        return error.returncode
    return 0
