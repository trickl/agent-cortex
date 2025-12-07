"""Helpers for inspecting and reusing persisted Java plan artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .java_planner import JavaPlanResult


_METADATA_FILENAME = "plan_metadata.json"


@dataclass(frozen=True)
class CachedPlan:
    """Represents a previously compiled plan recovered from disk."""

    prompt_hash: str
    attempt_number: int
    plan_dir: Path
    classes_dir: Path
    plan: JavaPlanResult
    metadata: Dict[str, Any]

    @property
    def plan_source(self) -> str:
        return self.plan.plan_source


class PlanCache:
    """Utility wrapper for enumerating cached plan attempts."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def prompt_dir(self, prompt_hash: str) -> Path:
        return self._root / prompt_hash

    def highest_attempt(self, prompt_hash: str) -> int:
        prompt_dir = self.prompt_dir(prompt_hash)
        if not prompt_dir.exists():
            return 0
        numbers = [entry.name for entry in prompt_dir.iterdir() if entry.is_dir()]
        parsed: List[int] = []
        for name in numbers:
            try:
                parsed.append(int(name))
            except ValueError:
                continue
        return max(parsed, default=0)

    def load(
        self,
        prompt_hash: str,
        *,
        stub_hash: Optional[str],
        stub_class_name: Optional[str],
    ) -> Optional[CachedPlan]:
        prompt_dir = self.prompt_dir(prompt_hash)
        if not prompt_dir.exists():
            return None
        attempt_dirs = self._sorted_attempt_dirs(prompt_dir)
        for attempt_number, attempt_dir in attempt_dirs:
            clean_path = attempt_dir / "clean"
            classes_dir = attempt_dir / "classes"
            if not clean_path.exists() or not classes_dir.exists():
                continue
            if not any(classes_dir.rglob("*.class")):
                continue
            metadata = self._read_metadata(attempt_dir)
            if not self._tool_metadata_matches(metadata, stub_hash, stub_class_name):
                continue
            plan_path = attempt_dir / "Plan.java"
            if not plan_path.exists():
                continue
            plan_source = plan_path.read_text(encoding="utf-8")
            plan = self._plan_from_metadata(metadata, plan_source, prompt_hash)
            if plan is None:
                continue
            return CachedPlan(
                prompt_hash=prompt_hash,
                attempt_number=attempt_number,
                plan_dir=attempt_dir,
                classes_dir=classes_dir,
                plan=plan,
                metadata=metadata,
            )
        return None

    # ------------------------------------------------------------------
    # Internal helpers

    def _sorted_attempt_dirs(self, prompt_dir: Path) -> List[tuple[int, Path]]:
        attempt_dirs: List[tuple[int, Path]] = []
        for entry in prompt_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                attempt = int(entry.name)
            except ValueError:
                continue
            attempt_dirs.append((attempt, entry))
        attempt_dirs.sort(key=lambda item: item[0], reverse=True)
        return attempt_dirs

    def _read_metadata(self, attempt_dir: Path) -> Dict[str, Any]:
        metadata_path = attempt_dir / _METADATA_FILENAME
        if not metadata_path.exists():
            return {}
        try:
            content = metadata_path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _plan_from_metadata(
        self,
        metadata: Dict[str, Any],
        plan_source: str,
        prompt_hash: str,
    ) -> Optional[JavaPlanResult]:
        plan_id = metadata.get("plan_id") or prompt_hash
        raw_response = metadata.get("raw_response") or {}
        if not isinstance(raw_response, dict):
            raw_response = {}
        prompt_messages = metadata.get("prompt_messages") or []
        if not isinstance(prompt_messages, list):
            prompt_messages = []
        plan_metadata = metadata.get("plan_metadata") or {}
        if not isinstance(plan_metadata, dict):
            plan_metadata = {}
        stored_prompt_hash = metadata.get("prompt_hash") or prompt_hash
        try:
            return JavaPlanResult(
                plan_id=str(plan_id),
                plan_source=plan_source,
                raw_response=dict(raw_response),
                prompt_messages=list(prompt_messages),
                metadata=dict(plan_metadata),
                prompt_hash=str(stored_prompt_hash),
            )
        except Exception:
            return None

    @staticmethod
    def _tool_metadata_matches(
        metadata: Dict[str, Any],
        expected_hash: Optional[str],
        expected_class: Optional[str],
    ) -> bool:
        stored_hash = metadata.get("tool_stub_hash")
        stored_class = metadata.get("tool_stub_class_name")
        if stored_class:
            if expected_class is None or expected_class != stored_class:
                return False
        elif expected_class:
            return False
        if stored_hash:
            if expected_hash is None or expected_hash != stored_hash:
                return False
        elif expected_hash:
            return False
        return True


def compute_stub_hash(source: Optional[str]) -> Optional[str]:
    """Return a stable hash for the provided tool stub source."""

    if source is None:
        return None
    normalized = source.strip()
    if not normalized:
        return None
    digest = hashlib.sha256()
    digest.update(normalized.encode("utf-8"))
    return digest.hexdigest()


__all__ = ["CachedPlan", "PlanCache", "compute_stub_hash"]
