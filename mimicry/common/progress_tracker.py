"""JSON progress tracker with snapshot and bookmark helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import PIPELINE_STEPS
from .state_getters import (
    get_mimic_art_state,
    get_original_art_state,
    get_protected_art_state,
    get_robust_samples_state,
)


def now_iso() -> str:
    """Return a UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ProgressTracker:
    """Track pipeline progress and avoid redundant work via JSON bookmarks."""

    tracker_path: Path
    data_root: Path
    min_samples_per_artist: int = 20
    _state: dict[str, Any] = field(default_factory=dict)

    def load(self) -> dict[str, Any]:
        """Load tracker JSON from disk, or initialize an empty tracker."""
        if self.tracker_path.exists():
            with self.tracker_path.open("r", encoding="utf-8") as handle:
                self._state = json.load(handle)
        else:
            self._state = self._default_state()
        return self._state

    def save(self) -> None:
        """Persist the tracker state to disk."""
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        with self.tracker_path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _default_state(self) -> dict[str, Any]:
        """Create a default tracker payload."""
        return {
            "metadata": {
                "created_at": now_iso(),
                "last_updated_at": now_iso(),
                "data_root": str(self.data_root),
                "min_samples_per_artist": self.min_samples_per_artist,
            },
            "step_status": {step: "not_started" for step in PIPELINE_STEPS},
            "bookmarks": {step: {} for step in PIPELINE_STEPS},
            "snapshot": {},
        }

    def refresh_snapshot(self) -> dict[str, Any]:
        """Refresh filesystem-derived progress metrics and inferred step status."""
        if not self._state:
            self.load()

        snapshot = {
            "captured_at": now_iso(),
            "original_art": get_original_art_state(
                data_root=self.data_root,
                min_samples_per_artist=self.min_samples_per_artist,
            ),
            "protected_art": get_protected_art_state(data_root=self.data_root),
            "robust_samples": get_robust_samples_state(data_root=self.data_root),
            "mimic_art": get_mimic_art_state(data_root=self.data_root),
        }
        self._state["snapshot"] = snapshot
        self._state["step_status"] = infer_step_status(snapshot=snapshot)
        self._state["metadata"]["last_updated_at"] = now_iso()
        self._state["metadata"]["min_samples_per_artist"] = self.min_samples_per_artist
        self._state["metadata"]["data_root"] = str(self.data_root)
        self.save()
        return self._state

    def has_bookmark(self, step_name: str, bookmark_key: str) -> bool:
        """Return whether a bookmark key already exists for the given step."""
        if not self._state:
            self.load()
        return bookmark_key in self._state.get("bookmarks", {}).get(step_name, {})

    def set_bookmark(self, step_name: str, bookmark_key: str, payload: dict[str, Any]) -> None:
        """Create or update a bookmark record for a pipeline step."""
        if not self._state:
            self.load()
        if step_name not in self._state["bookmarks"]:
            self._state["bookmarks"][step_name] = {}
        self._state["bookmarks"][step_name][bookmark_key] = {
            **payload,
            "updated_at": now_iso(),
        }
        self._state["metadata"]["last_updated_at"] = now_iso()
        self.save()


def infer_step_status(snapshot: dict[str, Any]) -> dict[str, str]:
    """Infer high-level step status from current filesystem snapshot."""
    statuses: dict[str, str] = {step: "not_started" for step in PIPELINE_STEPS}

    original = snapshot["original_art"]
    protected = snapshot["protected_art"]
    robust = snapshot["robust_samples"]
    mimic = snapshot["mimic_art"]

    if original["totals"]["artists"] > 0:
        statuses["step_1_collect_original_art"] = (
            "complete" if original["totals"]["artists_below_threshold"] == 0 else "in_progress"
        )

    expected_per_method = original["totals"]["images"]
    if protected["totals"]["images"] > 0:
        statuses["step_2_apply_protections"] = "in_progress"
        if expected_per_method > 0 and all(
            method_state["exists"] and method_state["image_count"] >= expected_per_method
            for method_state in protected["methods"].values()
        ):
            statuses["step_2_apply_protections"] = "complete"

    if robust["totals"]["images"] > 0:
        statuses["step_3_fine_tune_sd35"] = "in_progress"

    if mimic["image_count"] > 0:
        statuses["step_4_generate_mimic_art"] = "in_progress"

    return statuses
