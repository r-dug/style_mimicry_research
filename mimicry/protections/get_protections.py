#!/usr/bin/env python3
"""Bootstrap and optionally fetch protection tool sources."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICRY_DIR = REPO_ROOT / "mimicry"
if str(MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICRY_DIR))

from protections.catalog import PROTECTION_SOURCES, get_method_list, to_manifest_entry  # noqa: E402


def now_iso() -> str:
    """Return UTC ISO-8601 timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for source bootstrap/fetch operations."""
    parser = argparse.ArgumentParser(description="Prepare protection tool sources for Step 2.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Protection method to process (repeatable). Defaults to all canonical methods.",
    )
    parser.add_argument(
        "--tools-root",
        type=Path,
        default=Path("mimicry/protections/tools"),
        help="Directory containing cloned/downloaded protection sources.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("mimicry/protections/tools/protection_sources.json"),
        help="Manifest file to write source bootstrap status.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Clone/fetch repositories where applicable.",
    )
    parser.add_argument(
        "--ref",
        action="append",
        default=[],
        help="Override git ref per method in form method=ref (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying sources.",
    )
    return parser.parse_args()


def parse_ref_overrides(items: list[str]) -> dict[str, str]:
    """Parse method=ref override arguments."""
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --ref format: {item}. Expected method=ref.")
        method, ref = item.split("=", 1)
        method = method.strip()
        ref = ref.strip()
        if not method or not ref:
            raise ValueError(f"Invalid --ref format: {item}. Expected method=ref.")
        overrides[method] = ref
    return overrides


def run_git(args: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    """Run a git command unless in dry-run mode."""
    cmd = ["git", *args]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch_method_source(
    method: str,
    source_dir: Path,
    repo_url: str,
    ref: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Clone or update a git repository for one protection method."""
    result: dict[str, Any] = {
        "method": method,
        "source_dir": str(source_dir),
        "repo_url": repo_url,
        "ref": ref,
        "fetched": False,
        "status": "not_fetched",
    }

    if repo_url.startswith("https://glaze.cs.uchicago.edu/"):
        result["status"] = "manual_source_required"
        return result

    git_dir = source_dir / ".git"
    if git_dir.exists():
        run_git(["fetch", "--all", "--tags"], cwd=source_dir, dry_run=dry_run)
        run_git(["checkout", ref], cwd=source_dir, dry_run=dry_run)
        run_git(["pull", "--ff-only"], cwd=source_dir, dry_run=dry_run)
        result["status"] = "updated"
    else:
        if dry_run:
            print(f"[dry-run] git clone --branch {ref} {repo_url} {source_dir}")
        else:
            source_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--branch", ref, repo_url, str(source_dir)],
                check=True,
            )
        result["status"] = "cloned"
    result["fetched"] = True
    return result


def main() -> int:
    """Bootstrap/fetch method sources and write manifest."""
    args = parse_args()
    methods = get_method_list(args.method)
    ref_overrides = parse_ref_overrides(args.ref)

    manifest: dict[str, Any] = {
        "generated_at": now_iso(),
        "tools_root": str(args.tools_root),
        "fetch_requested": args.fetch,
        "methods": {},
    }

    for method in methods:
        source_meta = PROTECTION_SOURCES[method]
        ref = ref_overrides.get(method, source_meta.default_ref)
        source_dir = args.tools_root / method
        entry: dict[str, Any] = {
            **to_manifest_entry(source_meta),
            "resolved_ref": ref,
            "source_dir": str(source_dir),
            "local_source_exists": source_dir.exists(),
            "actions": [],
        }

        if args.fetch:
            fetch_result = fetch_method_source(
                method=method,
                source_dir=source_dir,
                repo_url=source_meta.repo_url,
                ref=ref,
                dry_run=args.dry_run,
            )
            entry["actions"].append(fetch_result)
            entry["local_source_exists"] = source_dir.exists() or args.dry_run
        else:
            entry["actions"].append(
                {
                    "status": "manifest_only",
                    "message": "Run with --fetch to clone/update sources.",
                }
            )

        manifest["methods"][method] = entry

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote protection manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

