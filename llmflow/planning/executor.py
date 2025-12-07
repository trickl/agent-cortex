"""Runtime bridge that executes Java plans via the Python syscall runtime."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from llmflow.runtime.errors import ToolError
from llmflow.runtime.syscall_registry import SyscallRegistry

from .java_plan_analysis import JavaPlanGraph, analyze_java_plan
from .runtime.interpreter import ExecutionTracer, PlanInterpreter, PlanRuntimeError
from .runtime.parser import PlanParseError, parse_java_plan
from .runtime.validator import PlanValidator, ValidationError, ValidationIssue


class PlanExecutor:
    """Execute Java planner output against a syscall registry."""

    def __init__(self, registry: SyscallRegistry) -> None:
        if registry is None:
            raise ValueError("PlanExecutor requires a SyscallRegistry instance")
        self._registry = registry

    def execute_from_string(
        self,
        plan_source: str,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        goal_summary: Optional[str] = None,
        deferred_metadata: Optional[Dict[str, Any]] = None,
        deferred_constraints: Optional[Sequence[str]] = None,
        tool_stub_class_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse, validate, and execute ``plan_source``."""

        return self._execute(
            plan_source,
            capture_trace=capture_trace,
            metadata=metadata,
            goal_summary=goal_summary,
            deferred_metadata=deferred_metadata,
            deferred_constraints=deferred_constraints,
            tool_stub_class_name=tool_stub_class_name,
        )

    def execute(self, plan_source: str, **kwargs) -> Dict[str, Any]:
        """Adapter so :class:`PlanExecutor` matches the :class:`PlanRunner` API."""

        return self.execute_from_string(plan_source, **kwargs)

    # ------------------------------------------------------------------

    def _execute(
        self,
        plan_source: str,
        *,
        capture_trace: bool,
        metadata: Optional[Dict[str, Any]],
        goal_summary: Optional[str],
        deferred_metadata: Optional[Dict[str, Any]],
        deferred_constraints: Optional[Sequence[str]],
        tool_stub_class_name: Optional[str],
    ) -> Dict[str, Any]:
        metadata_payload = self._build_metadata(
            metadata,
            goal_summary=goal_summary,
            deferred_metadata=deferred_metadata,
            deferred_constraints=deferred_constraints,
        )
        tracer = ExecutionTracer(enabled=capture_trace)

        try:
            plan = parse_java_plan(
                plan_source,
                tool_stub_class_name=tool_stub_class_name,
            )
        except PlanParseError as exc:
            return self._format_failure(
                "parse_error",
                str(exc),
                metadata_payload,
                tracer,
            )

        validator = PlanValidator(available_syscalls=self._registry.to_dict().keys())
        try:
            validator.validate(plan)
        except ValidationError as exc:
            return self._format_validation_failure(exc.issues, metadata_payload, tracer)

        graph_dict = self._analyze_plan(plan_source, tool_stub_class_name, metadata_payload)

        interpreter = PlanInterpreter(
            plan,
            registry=self._registry,
            tracer=tracer,
        )
        try:
            return_value = interpreter.run()
        except ToolError as exc:
            return self._format_failure("tool_error", str(exc), metadata_payload, tracer, graph=graph_dict)
        except PlanRuntimeError as exc:
            return self._format_failure("runtime_error", str(exc), metadata_payload, tracer, graph=graph_dict)
        except Exception as exc:  # pragma: no cover - defensive guard
            return self._format_failure(
                "unexpected_error",
                str(exc),
                metadata_payload,
                tracer,
                graph=graph_dict,
            )

        return {
            "success": True,
            "errors": [],
            "metadata": metadata_payload,
            "trace": tracer.as_list(),
            "graph": graph_dict,
            "return_value": return_value,
        }

    # ------------------------------------------------------------------
    # Formatting helpers

    @staticmethod
    def _build_metadata(
        metadata: Optional[Dict[str, Any]],
        *,
        goal_summary: Optional[str],
        deferred_metadata: Optional[Dict[str, Any]],
        deferred_constraints: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        payload = dict(metadata or {})
        if goal_summary is not None:
            payload.setdefault("goal_summary", goal_summary)
        if deferred_metadata is not None:
            payload.setdefault("deferred_metadata", dict(deferred_metadata))
        if deferred_constraints is not None:
            payload.setdefault("deferred_constraints", list(deferred_constraints))
        return payload

    @staticmethod
    def _format_failure(
        error_type: str,
        message: str,
        metadata: Dict[str, Any],
        tracer: ExecutionTracer,
        *,
        graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "errors": [
                {
                    "type": error_type,
                    "message": message,
                }
            ],
            "metadata": metadata,
            "trace": tracer.as_list(),
            "graph": graph,
        }

    @staticmethod
    def _format_validation_failure(
        issues: Iterable[ValidationIssue],
        metadata: Dict[str, Any],
        tracer: ExecutionTracer,
    ) -> Dict[str, Any]:
        errors = [
            {
                "type": "validation_error",
                "message": issue.message,
                "line": issue.line,
                "column": issue.column,
                "function": issue.function,
            }
            for issue in issues
        ]
        return {
            "success": False,
            "errors": errors,
            "metadata": metadata,
            "trace": tracer.as_list(),
            "graph": None,
        }

    def _analyze_plan(
        self,
        plan_source: str,
        tool_stub_class_name: Optional[str],
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            graph = analyze_java_plan(
                plan_source,
                tool_stub_class_name=tool_stub_class_name,
            )
        except Exception:  # pragma: no cover - best-effort metadata
            return None
        self._attach_graph_metadata(metadata, graph)
        return graph.to_dict()

    @staticmethod
    def _attach_graph_metadata(metadata: Dict[str, Any], graph: JavaPlanGraph) -> None:
        functions = graph.functions
        metadata.setdefault("functions", len(functions))
        metadata.setdefault("function_names", [fn.name for fn in functions])
        metadata.setdefault(
            "tool_call_count",
            sum(len(fn.tool_calls) for fn in functions),
        )


__all__ = ["PlanExecutor"]
