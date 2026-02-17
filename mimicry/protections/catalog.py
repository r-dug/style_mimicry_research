"""Protection tool catalog and metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from common.constants import PROTECTED_METHODS


@dataclass(frozen=True)
class ProtectionSource:
    """Metadata for one protection method source repository."""

    method: str
    repo_url: str
    default_ref: str
    notes: str


PROTECTION_SOURCES: dict[str, ProtectionSource] = {
    "glaze": ProtectionSource(
        method="glaze",
        repo_url="https://glaze.cs.uchicago.edu/",
        default_ref="n/a",
        notes="No public source repository; use official executable/web tooling.",
    ),
    "mist": ProtectionSource(
        method="mist",
        repo_url="https://github.com/psyker-team/mist-v2",
        default_ref="main",
        notes="Public repository.",
    ),
    "anti_dreambooth": ProtectionSource(
        method="anti_dreambooth",
        repo_url="https://github.com/VinAIResearch/Anti-DreamBooth",
        default_ref="main",
        notes="Public repository.",
    ),
    "style_guard": ProtectionSource(
        method="style_guard",
        repo_url="https://github.com/PolyLiYJ/StyleGuard",
        default_ref="main",
        notes="Public repository.",
    ),
}


def get_method_list(methods: list[str] | None = None) -> list[str]:
    """Return validated protection methods in deterministic order."""
    if not methods:
        return list(PROTECTED_METHODS)

    unknown = sorted(set(methods) - set(PROTECTED_METHODS))
    if unknown:
        raise ValueError(f"Unknown protection method(s): {', '.join(unknown)}")
    return sorted(dict.fromkeys(methods))


def to_manifest_entry(source: ProtectionSource) -> dict[str, Any]:
    """Convert source metadata to a JSON-serializable dict."""
    return asdict(source)

