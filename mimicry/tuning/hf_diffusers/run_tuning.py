#!/usr/bin/env python3
"""HF Diffusers DreamBooth LoRA SD3.5 tuning runner wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

# runner_common lives in mimicry/protections/
PROTECTIONS_DIR = Path(__file__).resolve().parents[2] / "protections"
if str(PROTECTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(PROTECTIONS_DIR))

from runner_common import run_method_runner  # noqa: E402

if __name__ == "__main__":
    default_config = Path(__file__).resolve().parent / "runner_config.json"
    raise SystemExit(run_method_runner(method="hf_diffusers", default_config_path=default_config))
