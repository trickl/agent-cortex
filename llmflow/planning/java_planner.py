"""Utilities for prompting the LLM to emit Java plans."""
from __future__ import annotations

import json
import logging
import re
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

try:  # pragma: no cover - guard against optional dependency changes
    from instructor.core.exceptions import InstructorRetryException
except ImportError:  # pragma: no cover
    InstructorRetryException = None  # type: ignore[assignment]

from llmflow.llm_client import LLMClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SPEC_PATH = _PROJECT_ROOT / "planning" / "java_planning.md"
_PLANNER_TOOL_NAME = "define_java_plan"

logger = logging.getLogger(__name__)

_CLASS_DECL_PATTERN = re.compile(r"^\s*(?:public\s+)?class\s+\w+", re.MULTILINE)
_CODE_FENCE_PATTERN = re.compile(r"```(?P<lang>[^\n`]*)\n", re.MULTILINE)
_MARKDOWN_PREFIX_PATTERN = re.compile(
    r"^(?:#{1,6}\s+|>{1,}\s+|[-*+]\s+|\d+\.\s+|`{1,3}$)",
)
_LIKELY_JAVA_PREFIX_PATTERN = re.compile(
    r"^(?:package\s+|import\s+|(?:public|protected|private|abstract|final|static)\b|class\s+|interface\s+|enum\s+|record\s+|@|/\*|//|\*)",
)


def _normalize_java_source(source: str) -> str:
    """Normalize Java source and ensure it declares a top-level class."""

    stripped = source.strip()
    candidate = _extract_java_candidate(stripped)
    if not _CLASS_DECL_PATTERN.search(candidate):
        raise ValueError("Java payload must declare a top-level class.")
    return candidate.strip()


def _extract_java_candidate(source: str) -> str:
    block = _extract_code_block(source)
    if block:
        cleaned = _strip_trailing_backticks(block)
        if cleaned:
            return cleaned

    trimmed = _trim_non_java_prefix(source)
    trimmed = _strip_trailing_backticks(trimmed)
    return trimmed or source


def _extract_code_block(source: str) -> Optional[str]:
    candidates: List[Dict[str, Any]] = []
    for match in _CODE_FENCE_PATTERN.finditer(source):
        lang = (match.group("lang") or "").strip().lower()
        block_start = match.end()
        block_end = source.find("```", block_start)
        closed = True
        if block_end == -1:
            block_end = len(source)
            closed = False
        body = source[block_start:block_end].strip()
        if not body:
            continue
        candidates.append(
            {
                "lang": lang,
                "body": body,
                "closed": closed,
                "start": match.start(),
            }
        )

    if not candidates:
        return None

    def _score(candidate: Dict[str, Any]) -> Tuple[int, int, int]:
        lang_score = 1 if "java" in candidate["lang"] else 0
        closed_score = 1 if candidate["closed"] else 0
        return (lang_score, closed_score, candidate["start"])

    best = max(candidates, key=_score)
    return best["body"]


def _trim_non_java_prefix(source: str) -> str:
    lines = source.splitlines()
    result: List[str] = []
    dropping = True
    for line in lines:
        stripped = line.lstrip()
        if dropping:
            if not stripped:
                continue
            if stripped.startswith("```"):
                continue
            if _MARKDOWN_PREFIX_PATTERN.match(stripped):
                continue
            if _LIKELY_JAVA_PREFIX_PATTERN.match(stripped):
                dropping = False
                result.append(line)
                continue
            # Skip any other non-Java preamble lines.
            continue
        result.append(line)
    return "\n".join(result).lstrip()


def _strip_trailing_backticks(source: str) -> str:
    lines = source.splitlines()
    while lines and lines[-1].strip() in {"```", "``", "`"}:
        lines.pop()
    return "\n".join(lines).rstrip()


def _extract_tool_call_count(exc: Exception) -> Optional[int]:
    """Best-effort extraction of tool call count from Instructor retries."""

    if InstructorRetryException is None:
        return None
    if not isinstance(exc, InstructorRetryException):
        return None
    attempts = getattr(exc, "failed_attempts", None)
    if not attempts:
        return None
    for attempt in attempts:
        completion = getattr(attempt, "completion", None)
        if not completion:
            continue
        choices = getattr(completion, "choices", None)
        if not choices:
            continue
        message = getattr(choices[0], "message", None)
        if not message:
            continue
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None:
            return 0
        return len(tool_calls)
    return None


def _summarize_structured_failure(exc: Exception) -> Optional[str]:
    """Return a user-friendly explanation for structured generation failures."""

    count = _extract_tool_call_count(exc)
    if count == 0:
        return "Structured Java plan request produced no tool calls."
    if count and count != 1:
        return f"Structured Java plan request produced {count} tool calls; expected exactly one."

    message = str(exc)
    if "Instructor does not support multiple tool calls" in message:
        return "Structured Java plan request did not yield exactly one tool call."
    if message:
        return f"Structured Java plan request failed: {message}"
    return None


class JavaPlanningError(RuntimeError):
    """Raised when Java plan synthesis fails."""


@dataclass
class JavaPlanRequest:
    """Inputs that describe what the planner should generate."""

    task: str
    goals: Sequence[str] = field(default_factory=list)
    context: Optional[str] = None
    tool_names: Sequence[str] = field(default_factory=list)
    tool_schemas: Sequence[Dict[str, Any]] = field(default_factory=list)
    additional_constraints: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    include_deferred_guidance: bool = True
    tool_stub_source: Optional[str] = None
    tool_stub_class_name: Optional[str] = None
    prior_plan_source: Optional[str] = None
    compile_error_report: Optional[str] = None
    refinement_iteration: int = 0


@dataclass
class JavaPlanResult:
    """Structured result returned by :class:`JavaPlanner`."""

    plan_id: str
    plan_source: str
    raw_response: Dict[str, Any]
    prompt_messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class _PlannerToolPayload(BaseModel):
    notes: Optional[str] = Field(
        default=None,
        description="Optional commentary about assumptions, risks, or TODOs.",
    )
    java: str = Field(
        ..., description="Complete Java source code containing exactly one top-level class."
    )

    @field_validator("java")
    @classmethod
    def _ensure_plan_block(cls, value: str) -> str:
        try:
            return _normalize_java_source(value)
        except ValueError as exc:  # pragma: no cover - validated downstream
            raise ValueError(str(exc)) from exc

    @field_validator("notes", mode="before")
    @classmethod
    def _normalize_notes(cls, value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            if not normalized:
                return None
            return "\n\n".join(normalized)
        return str(value)


class JavaPlanner:
    """High-level helper that asks the LLM to emit a Java plan."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        specification: Optional[str] = None,
        specification_path: Optional[Path] = None,
    ):
        self._llm_client = llm_client
        self._specification = self._load_specification(specification, specification_path)
        self._planner_tool_schema = self._build_planner_tool_schema()
        self._planner_tool_choice = {
            "type": "function",
            "function": {"name": _PLANNER_TOOL_NAME},
        }

    def generate_plan(self, request: JavaPlanRequest) -> JavaPlanResult:
        messages = self._build_messages(request)
        plan_source: str
        raw_response: Dict[str, Any]
        notes: Optional[str] = None
        request_start = time.perf_counter()
        retries = (
            getattr(self._llm_client.provider, "max_retries", None)
            or getattr(self._llm_client.provider, "default_retries", None)
            or 0
        )
        status_prefix = "[JavaPlanner]"
        print(
            f"{status_prefix} Requesting structured Java plan (tools={request.tool_names or ['none']}, retries={retries})",
            flush=True,
        )
        logger.info(
            "%s Requesting structured Java plan (tools=%s, retries=%s)",
            status_prefix,
            request.tool_names or ["none"],
            retries,
        )
        try:
            payload = self._llm_client.structured_generate(
                messages=messages,
                response_model=_PlannerToolPayload,
                tools=[self._planner_tool_schema],
                tool_choice=self._planner_tool_choice,
            )
        except Exception as exc:  # pragma: no cover - provider dependent
            elapsed = time.perf_counter() - request_start
            print(
                f"{status_prefix} Structured plan request failed after {elapsed:.1f}s: {exc}",
                flush=True,
            )
            logger.warning(
                "%s Structured plan request failed after %.1fs: %s",
                status_prefix,
                elapsed,
                exc,
            )
            friendly = _summarize_structured_failure(exc)
            if friendly:
                logger.warning("%s Falling back to plain-text parsing.", friendly)
            else:
                logger.warning(
                    "Structured Java plan generation failed; attempting plain-text fallback.",
                    exc_info=exc,
                )
            plan_source, raw_response, notes = self._generate_plain_plan(messages)
        else:
            elapsed = time.perf_counter() - request_start
            print(
                f"{status_prefix} Structured plan ready in {elapsed:.1f}s (notes={bool(payload.notes)})",
                flush=True,
            )
            logger.info(
                "%s Structured plan ready in %.1fs (notes=%s)",
                status_prefix,
                elapsed,
                bool(payload.notes),
            )
            plan_source = payload.java.strip()
            raw_response = payload.model_dump()
            if payload.notes:
                notes = payload.notes.strip()

        plan_id = str(request.metadata.get("plan_id") or uuid.uuid4())
        metadata = dict(request.metadata)
        metadata.setdefault("allowed_tools", sorted(request.tool_names))
        if notes:
            metadata["planner_notes"] = notes
        return JavaPlanResult(
            plan_id=plan_id,
            plan_source=plan_source,
            raw_response=raw_response,
            prompt_messages=messages,
            metadata=metadata,
        )

    def _generate_plain_plan(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Fallback path when structured tool calls are unavailable."""

        response = self._llm_client.generate(
            messages=messages,
            tools=[self._planner_tool_schema],
            tool_choice=self._planner_tool_choice,
        )
        if not isinstance(response, dict):
            raise JavaPlanningError("Planner returned an unexpected response format.")

        notes: Optional[str] = None
        tool_calls = response.get("tool_calls") or []
        if len(tool_calls) == 1:
            call = tool_calls[0]
            function_meta = call.get("function", {})
            if function_meta.get("name") == _PLANNER_TOOL_NAME:
                arguments = function_meta.get("arguments") or "{}"
                try:
                    payload_data = json.loads(arguments)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise JavaPlanningError(
                        "Planner tool call arguments were invalid JSON."
                    ) from exc
                java_source = payload_data.get("java")
                if not java_source:
                    raise JavaPlanningError("Planner tool call did not include Java source.")
                try:
                    normalized = _normalize_java_source(java_source)
                except ValueError as exc:
                    raise JavaPlanningError(f"Planner returned invalid Java: {exc}") from exc
                raw_notes = payload_data.get("notes")
                if raw_notes is not None:
                    notes = _PlannerToolPayload._normalize_notes(raw_notes)
                return normalized, response, notes

        content = response.get("content")
        if content is None or not str(content).strip():
            raise JavaPlanningError("Planner returned empty content.")
        try:
            normalized = _normalize_java_source(str(content))
        except ValueError as exc:
            raise JavaPlanningError(f"Planner returned invalid Java: {exc}") from exc
        return normalized, response, None

    def _build_messages(self, request: JavaPlanRequest) -> List[Dict[str, Any]]:
        has_tools = bool(request.tool_names)
        constraint_lines = self._build_constraints(request, has_tools)
        system_content = self._build_system_message(
            request.tool_stub_source,
            request.tool_stub_class_name,
        )
        user_content = self._build_user_message(request, constraint_lines)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _build_system_message(
        self,
        tool_stub_source: Optional[str] = None,
        tool_stub_class_name: Optional[str] = None,
    ) -> str:
        header = (
            "You are the Java plan synthesizer."
            " Produce a single Java class that fully solves the user's task."
            " Refer to the define_java_plan tool description for the full specification,"
            " calling that tool exactly once when your model supports tool calls."
            " If tools are unavailable, respond with the raw Java source only and do not"
            " emit explanations."
        )
        schema_guidance = textwrap.dedent(
            """
            When the runtime requests structured output, you must emit a single JSON object matching:
            {
              "java": "public class Plan { ... }",
              "notes": "Optional commentary about risks or TODOs (omit or null when unused)"
            }
            Do not introduce additional keys, arrays, or wrapper text around this object.
            """
        ).strip()
        tools_block = json.dumps(
            {
                "available_tools": [self._planner_tool_schema["function"]],
                "instructions": (
                    "Call define_java_plan exactly once when possible; otherwise return the"
                    " Java source as plain assistant text."
                ),
            },
            indent=2,
        )
        message = (
            f"{header}\n\n{schema_guidance}\n\n<available_tools>\n{tools_block}\n</available_tools>"
        ).strip()

        if tool_stub_source:
            stub_intro = [
                "Tool stub reference: Use the provided Java class when calling tools.",
            ]
            if tool_stub_class_name:
                stub_intro.append(
                    f"Call the static methods defined on `{tool_stub_class_name}` instead of inventing new signatures."
                )
            stub_intro_text = " ".join(stub_intro).strip()
            stub_block = textwrap.dedent(
                f"""
                {stub_intro_text}

                <tool_stubs>
                {tool_stub_source.strip()}
                </tool_stubs>
                """
            ).strip()
            message = f"{message}\n\n{stub_block}".strip()

        return message

    def _build_user_message(
        self,
        request: JavaPlanRequest,
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

        lines.append("Available planning tools:")
        tool_entries = _summarize_tools(request.tool_schemas, request.tool_names)
        if tool_entries:
            lines.extend(tool_entries)
        else:
            lines.append("- (none registered)")
        lines.append("")

        if request.tool_stub_class_name:
            lines.append(
                f"Use the static methods on {request.tool_stub_class_name} to invoke these tools; do not invent other APIs."
            )
            lines.append("")

        lines.append("Constraints:")
        for rule in constraint_lines:
            lines.append(f"- {rule}")
        if request.additional_constraints:
            for rule in request.additional_constraints:
                lines.append(f"- {rule}")
        lines.append("")

        if request.prior_plan_source:
            lines.append("Previous plan attempt:")
            lines.append("```java")
            lines.append(request.prior_plan_source.strip())
            lines.append("```")
            lines.append("")

        if request.compile_error_report:
            lines.append("Compile diagnostics:")
            lines.append(request.compile_error_report.strip())
            lines.append("")
            lines.append(
                "Revise the plan to satisfy the Java planning specification and resolve each diagnostic above."
            )
            lines.append("")

        lines.append(
            "Output requirements: respond with only the Java source, preferably via the"
            f" {_PLANNER_TOOL_NAME} function when tool calls are supported."
        )
        return "\n".join(lines).strip()

    def _build_constraints(
        self,
        request: JavaPlanRequest,
        has_tools: bool,
    ) -> List[str]:
        stub_name = request.tool_stub_class_name or "PlanningToolStubs"
        constraints = [
            "Emit exactly one top-level Java class (any name) with helper methods and a main() entrypoint when needed.",
            f"Call tools exclusively via the `{stub_name}.<name>(...)` static helpers; never invent new APIs.",
            "Limit every helper body to seven statements and ensure each helper is more specific than its caller.",
            "Stick to the allowed statement types (variable declarations, assignments, helper/tool calls, if/else, enhanced for, try/catch, returns).",
            "Do not wrap the output in markdown; Java comments and imports are allowed but avoid prose explanations.",
        ]
        if not has_tools:
            constraints.append(
                "If no planning tools are available, describe diagnostic steps using logging and TODOs."
            )
        return constraints

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
        except OSError as exc:
            raise JavaPlanningError(f"Failed to load plan specification from {path}") from exc

    def _build_planner_tool_schema(self) -> Dict[str, Any]:
        description_lines = [
            "Return the final Java plan along with any helpful notes.",
            "Every response must comply with this specification:",
            self._specification,
        ]
        return {
            "type": "function",
            "function": {
                "name": _PLANNER_TOOL_NAME,
                "description": "\n\n".join(description_lines).strip(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "description": "Optional commentary about assumptions, risks, or TODOs.",
                        },
                        "java": {
                            "type": "string",
                            "description": "Complete Java source code containing a single top-level class.",
                        },
                    },
                    "required": ["java"],
                },
            },
        }


def _summarize_tools(
    tool_schemas: Sequence[Dict[str, Any]],
    fallback_names: Sequence[str],
) -> List[str]:
    entries: List[str] = []
    seen_names: set[str] = set()
    for schema in tool_schemas:
        if not isinstance(schema, dict):
            continue
        function_meta = schema.get("function") or {}
        name = function_meta.get("name")
        if not name or name in seen_names:
            continue
        description = function_meta.get("description") or "No description provided."
        entries.append(f"- {name}: {description}".strip())
        seen_names.add(name)
    if entries:
        return entries
    deduped_names = []
    for name in fallback_names:
        if name and name not in deduped_names:
            deduped_names.append(name)
    return [f"- {name}" for name in deduped_names]


__all__ = [
    "JavaPlanRequest",
    "JavaPlanResult",
    "JavaPlanner",
    "JavaPlanningError",
]
