#!/usr/bin/env python3
"""Anti-DreamBooth method runner wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

PROTECTIONS_DIR = Path(__file__).resolve().parents[1]
if str(PROTECTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(PROTECTIONS_DIR))

from runner_common import run_method_runner  # noqa: E402


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parent / "runner_config.json"
    raise SystemExit(run_method_runner(method="anti_dreambooth", default_config_path=default_config))

