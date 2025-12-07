"""Shared structures describing compiled plan artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PlanExecutionArtifacts:
    """Metadata describing where compiled plan outputs live on disk."""

    plan_id: str
    attempt_number: int
    prompt_hash: Optional[str]
    attempt_dir: Path
    classes_dir: Path
    plan_class_name: Optional[str]
    stub_source_path: Optional[Path] = None
    tool_stub_class_name: Optional[str] = None


__all__ = ["PlanExecutionArtifacts"]
