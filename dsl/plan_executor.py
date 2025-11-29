"""High-level execution harness for CPL plans."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from lark import Lark
from lark.exceptions import UnexpectedInput

from .cpl_inerpreter import (
    CPLInterpreter,
    CPLRuntimeError,
    DeferredExecutionOptions,
    ExecutionTracer,
    SyscallRegistry,
    ToolError,
    load_cpl_parser,
    parse_cpl,
)
from .cpl_validator import ValidationError
from .deferred_planner import DeferredFunctionPrompt

_DEFAULT_GRAMMAR = Path(__file__).with_name("cpl.lark")


@dataclass
class ExecutionError:
    type: str
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    function: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": self.type,
            "message": self.message,
        }
        if self.line is not None:
            data["line"] = self.line
        if self.column is not None:
            data["column"] = self.column
        if self.function is not None:
            data["function"] = self.function
        return data


@dataclass
class PlanExecutionResult:
    success: bool
    return_value: Any = None
    errors: List[ExecutionError] = field(default_factory=list)
    trace: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "return_value": self.return_value,
            "errors": [err.to_dict() for err in self.errors],
            "trace": self.trace,
            "metadata": self.metadata,
        }


class PlanExecutor:
    """Convenience wrapper that parses, validates, and executes CPL plans."""

    def __init__(
        self,
        registry: SyscallRegistry,
        *,
        grammar_path: Optional[str] = None,
        parser: Optional[Lark] = None,
        deferred_planner: Optional[Callable[[DeferredFunctionPrompt], str]] = None,
        deferred_options: Optional[DeferredExecutionOptions] = None,
        dsl_specification: Optional[str] = None,
        body_parser: Optional[Lark] = None,
    ):
        self.registry = registry
        if parser is not None:
            self.parser = parser
        else:
            grammar_file = Path(grammar_path) if grammar_path else _DEFAULT_GRAMMAR
            self.parser = load_cpl_parser(str(grammar_file))
        self.deferred_planner = deferred_planner
        self.deferred_options = deferred_options
        self.dsl_specification = dsl_specification
        self.body_parser = body_parser

    def execute_from_string(
        self,
        source: str,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = self._execute(source, capture_trace=capture_trace, extra_metadata=metadata or {})
        return result.to_dict()

    def execute_from_file(
        self,
        path: str | Path,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source = Path(path).read_text(encoding="utf-8")
        return self.execute_from_string(source, capture_trace=capture_trace, metadata=metadata)

    # ------------------------------------------------------------------

    def _execute(
        self,
        source: str,
        *,
        capture_trace: bool,
        extra_metadata: Dict[str, Any],
    ) -> PlanExecutionResult:
        tracer = ExecutionTracer(enabled=capture_trace)
        try:
            plan = parse_cpl(source, self.parser)
            interpreter = CPLInterpreter(
                plan,
                registry=self.registry,
                tracer=tracer,
                deferred_planner=self.deferred_planner,
                deferred_options=self.deferred_options,
                dsl_specification=self.dsl_specification,
                body_parser=self.body_parser,
            )
            return_value = interpreter.run()
            trace_payload = tracer.as_list() if capture_trace else None
            metadata = {
                "functions": len(plan.functions),
                "has_trace": capture_trace,
            }
            metadata.update(extra_metadata)
            return PlanExecutionResult(
                success=True,
                return_value=return_value,
                trace=trace_payload,
                metadata=metadata,
            )
        except ValidationError as exc:
            return self._error_result("validation_error", exc, tracer, capture_trace, extra_metadata)
        except UnexpectedInput as exc:
            return self._error_result("parse_error", exc, tracer, capture_trace, extra_metadata)
        except ToolError as exc:
            return self._error_result("tool_error", exc, tracer, capture_trace, extra_metadata)
        except CPLRuntimeError as exc:
            return self._error_result("runtime_error", exc, tracer, capture_trace, extra_metadata)
        except Exception as exc:  # pragma: no cover - safeguard
            return self._error_result("internal_error", exc, tracer, capture_trace, extra_metadata)

    def _error_result(
        self,
        error_type: str,
        exc: Exception,
        tracer: ExecutionTracer,
        capture_trace: bool,
        extra_metadata: Dict[str, Any],
    ) -> PlanExecutionResult:
        trace_payload = tracer.as_list() if capture_trace else None
        metadata = {
            "has_trace": capture_trace,
        }
        metadata.update(extra_metadata)
        errors = self._exception_to_errors(error_type, exc)
        return PlanExecutionResult(
            success=False,
            errors=errors,
            trace=trace_payload,
            metadata=metadata,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _exception_to_errors(error_type: str, exc: Exception) -> List[ExecutionError]:
        if isinstance(exc, ValidationError):
            return [
                ExecutionError(
                    type=error_type,
                    message=issue.message,
                    line=issue.line,
                    column=issue.column,
                    function=issue.function,
                )
                for issue in exc.issues
            ]
        if isinstance(exc, UnexpectedInput):
            return [
                ExecutionError(
                    type=error_type,
                    message=str(exc).strip(),
                    line=getattr(exc, "line", None),
                    column=getattr(exc, "column", None),
                )
            ]
        return [ExecutionError(type=error_type, message=str(exc))]


__all__ = [
    "ExecutionError",
    "PlanExecutionResult",
    "PlanExecutor",
]
