"""Utilities for prompting the LLM to emit Cortex Planning Language (CPL) plans."""
from __future__ import annotations

import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llmflow.llm_client import LLMClient

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SPEC_PATH = _PROJECT_ROOT / "dsl" / "planning.md"


class CPLPlanningError(RuntimeError):
    """Raised when CPL plan synthesis fails."""


@dataclass
class CPLPlanRequest:
    """Inputs that describe what the planner should generate."""

    task: str
    goals: Sequence[str] = field(default_factory=list)
    context: Optional[str] = None
    allowed_syscalls: Sequence[str] = field(default_factory=list)
    additional_constraints: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    include_deferred_guidance: bool = True


@dataclass
class CPLPlanResult:
    """Structured result returned by :class:`CPLPlanner`."""

    plan_id: str
    plan_source: str
    raw_response: Dict[str, Any]
    prompt_messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CPLPlanner:
    """High-level helper that asks the LLM to emit a CPL plan."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        dsl_specification: Optional[str] = None,
        dsl_spec_path: Optional[Path] = None,
    ):
        self._llm_client = llm_client
        self._dsl_specification = self._load_specification(dsl_specification, dsl_spec_path)

    def generate_plan(self, request: CPLPlanRequest) -> CPLPlanResult:
        """Invoke the LLM to create a CPL program for ``request``."""

        messages = self._build_messages(request)
        response = self._llm_client.generate(messages=messages, tools=None)
        plan_source = self._extract_plan_source(response)
        plan_id = str(request.metadata.get("plan_id") or uuid.uuid4())
        normalized_syscalls = self._normalize_allowed_syscalls(request.allowed_syscalls)
        metadata = dict(request.metadata)
        metadata.setdefault("allowed_syscalls", normalized_syscalls)
        return CPLPlanResult(
            plan_id=plan_id,
            plan_source=plan_source,
            raw_response=response,
            prompt_messages=messages,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Prompt construction helpers

    def _build_messages(self, request: CPLPlanRequest) -> List[Dict[str, Any]]:
        normalized_syscalls = self._normalize_allowed_syscalls(request.allowed_syscalls)
        constraint_lines = self._build_constraint_lines(request, normalized_syscalls)
        system_content = self._build_system_message()
        user_content = self._build_user_message(request, normalized_syscalls, constraint_lines)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _build_system_message(self) -> str:
        header = (
            "You are the Cortex Planning DSL synthesizer."
            " Produce a single CPL program that fully solves the user's task."
            " Obey the specification verbatim and do not emit explanations."
        )
        return f"{header}\n\n{self._dsl_specification}".strip()

    def _build_user_message(
        self,
        request: CPLPlanRequest,
        normalized_syscalls: Sequence[str],
        constraint_lines: Sequence[str],
    ) -> str:
        lines: List[str] = []
        lines.append("Task:")
        lines.append(textwrap.dedent(request.task).strip())
        lines.append("")

        if request.goals:
            lines.append("Goals:")
            for idx, goal in enumerate(request.goals, start=1):
                lines.append(f"{idx}. {goal}")
            lines.append("")

        if request.context:
            lines.append("Context:")
            lines.append(textwrap.dedent(request.context).strip())
            lines.append("")

        lines.append("Allowed syscalls:")
        if normalized_syscalls:
            for name in normalized_syscalls:
                lines.append(f"- {name}")
        else:
            lines.append("- (none registered)")
        lines.append("")

        lines.append("Constraints:")
        for rule in constraint_lines:
            lines.append(f"- {rule}")
        if request.additional_constraints:
            for rule in request.additional_constraints:
                lines.append(f"- {rule}")
        lines.append("")

        lines.append(
            "Output requirements: respond with only the CPL program inside a single `plan { ... }` block."
        )
        return "\n".join(lines).strip()

    def _build_constraint_lines(
        self,
        request: CPLPlanRequest,
        normalized_syscalls: Sequence[str],
    ) -> List[str]:
        constraints = [
            "Use only the provided syscall names.",
            "Keep each function focused on one sub-goal and under 7 statements.",
            "Annotate functions that require runtime context with @Deferred.",
            "Include a main() entrypoint that orchestrates the workflow.",
        ]
        if not normalized_syscalls:
            constraints.append(
                "If no syscalls are available, describe diagnostic steps using pure functions and logging."
            )
        if not request.include_deferred_guidance:
            constraints = [rule for rule in constraints if "@Deferred" not in rule]
        return constraints

    # ------------------------------------------------------------------
    # Helpers

    def _extract_plan_source(self, response: Dict[str, Any]) -> str:
        if not isinstance(response, dict):
            raise CPLPlanningError("LLM response must be a dict")
        role = response.get("role")
        content = response.get("content")
        if role != "assistant":
            raise CPLPlanningError(f"Unexpected LLM role '{role}'")
        if not isinstance(content, str):
            raise CPLPlanningError("LLM response did not contain textual content")
        plan_source = content.strip()
        if not plan_source:
            raise CPLPlanningError("LLM returned an empty plan")
        if not plan_source.lstrip().startswith("plan"):
            raise CPLPlanningError("LLM response did not start with a CPL plan")
        return plan_source

    @staticmethod
    def _normalize_allowed_syscalls(allowed: Sequence[str] | None) -> List[str]:
        if not allowed:
            return []
        deduped = {name.strip() for name in allowed if name and name.strip()}
        return sorted(deduped)

    @staticmethod
    def _load_specification(
        override_content: Optional[str],
        override_path: Optional[Path],
    ) -> str:
        if override_content:
            return override_content.strip()
        path = override_path or _DEFAULT_SPEC_PATH
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:  # pragma: no cover - environment specific
            raise CPLPlanningError(f"Failed to load CPL specification from {path}") from exc


__all__ = [
    "CPLPlanRequest",
    "CPLPlanResult",
    "CPLPlanner",
    "CPLPlanningError",
]
