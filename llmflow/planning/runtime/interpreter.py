"""Interpreter for Java-based plan ASTs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from llmflow.runtime.errors import ToolError
from llmflow.runtime.syscall_registry import SyscallRegistry

from .ast import (
    Assign,
    BinaryOp,
    CallExpr,
    Expr,
    ExprStmt,
    ForStmt,
    FunctionDef,
    IfStmt,
    ListLiteral,
    Literal,
    MapLiteral,
    Plan,
    ReturnStmt,
    Stmt,
    SyscallExpr,
    TryCatchStmt,
    VarDecl,
    VarRef,
)
from .validator import PlanValidator, ValidationError


class PlanRuntimeError(Exception):
    """Generic runtime error during plan execution."""


class ReturnSignal(Exception):
    """Internal control flow exception representing a return statement."""

    def __init__(self, value: Any):
        self.value = value


@dataclass
class Environment:
    parent: Optional["Environment"] = None

    def __post_init__(self):
        self.vars: Dict[str, Any] = {}

    def define(self, name: str, value: Any):
        if name in self.vars:
            raise PlanRuntimeError(f"Variable '{name}' already defined")
        self.vars[name] = value

    def set(self, name: str, value: Any):
        env = self._find_env(name)
        if env is None:
            raise PlanRuntimeError(f"Variable '{name}' not defined")
        env.vars[name] = value

    def get(self, name: str) -> Any:
        env = self._find_env(name)
        if env is None:
            raise PlanRuntimeError(f"Variable '{name}' not defined")
        return env.vars[name]

    def _find_env(self, name: str) -> Optional["Environment"]:
        env = self
        while env:
            if name in env.vars:
                return env
            env = env.parent
        return None


class ExecutionTracer:
    """Collects optional execution trace events emitted by the interpreter."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.enabled = enabled
        self.events: List[Dict[str, Any]] = []
        self._sink = sink

    def emit(self, event_type: str, **payload):
        if not self.enabled:
            return
        event = {"type": event_type, **payload}
        self.events.append(event)
        if self._sink:
            self._sink(event)

    def clear(self):
        self.events.clear()

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self.events)


class PlanInterpreter:
    """Executes a parsed plan using the provided syscall registry."""

    def __init__(
        self,
        plan: Plan,
        *,
        registry: Optional[SyscallRegistry] = None,
        syscalls: Optional[Dict[str, Callable[..., Any]]] = None,
        tracer: Optional[ExecutionTracer] = None,
    ):
        if registry is None and syscalls is None:
            raise ValueError("A syscall mapping or registry must be provided")
        if registry is not None and syscalls is not None:
            raise ValueError("Provide either 'registry' or 'syscalls', not both")
        self.plan = plan
        self.registry = registry or SyscallRegistry.from_mapping(syscalls or {})
        self.syscalls = self.registry.to_dict()
        self.tracer = tracer or ExecutionTracer(enabled=False)
        self._call_stack: List[str] = []
        self._validate_plan()

    def run(self) -> Any:
        if "main" not in self.plan.functions:
            self._runtime_error("No main() function defined", node=self.plan)
        self._trace("execution_start", entry_point="main")
        result = self.call_function("main", [], call_node=self.plan.functions["main"])
        self._trace("execution_end", entry_point="main", return_value=result)
        return result

    # ------------------------------------------------------------------
    # Function calls

    def call_function(self, name: str, args: List[Any], call_node: Optional[Expr] = None) -> Any:
        if name not in self.plan.functions:
            self._runtime_error(f"Function '{name}' not defined", node=call_node)
        fn = self.plan.functions[name]
        if len(args) != len(fn.params):
            self._runtime_error(
                f"Function '{name}' expects {len(fn.params)} args, got {len(args)}",
                node=call_node or fn,
            )

        env = Environment(parent=None)
        for param, value in zip(fn.params, args):
            try:
                env.define(param.name, value)
            except PlanRuntimeError as exc:
                self._runtime_error(str(exc), node=param)

        if fn.body is None:
            self._runtime_error(f"Function '{fn.name}' is missing a body", node=call_node or fn)
        body = fn.body
        self._call_stack.append(name)
        self._trace("function_enter", function=name, args=args, line=fn.line)
        result: Any = None
        error: Optional[str] = None
        try:
            self._exec_block(body, env)
        except ReturnSignal as rs:
            result = rs.value
        except Exception as exc:  # pragma: no cover - propagate
            error = str(exc)
            raise
        finally:
            self._trace("function_exit", function=name, return_value=result, error=error, line=fn.line)
            self._call_stack.pop()
        return result


    # ------------------------------------------------------------------
    # Statement execution

    def _exec_block(self, stmts: List[Stmt], env: Environment):
        for stmt in stmts:
            self._exec_stmt(stmt, env)

    def _exec_stmt(self, stmt: Stmt, env: Environment):
        if isinstance(stmt, VarDecl):
            value = self._eval_expr(stmt.expr, env)
            try:
                env.define(stmt.name, value)
            except PlanRuntimeError as exc:
                self._runtime_error(str(exc), node=stmt)
            self._trace("var_decl", name=stmt.name, value=value, line=stmt.line)
            return

        if isinstance(stmt, Assign):
            value = self._eval_expr(stmt.expr, env)
            try:
                env.set(stmt.name, value)
            except PlanRuntimeError as exc:
                self._runtime_error(str(exc), node=stmt)
            self._trace("assignment", name=stmt.name, value=value, line=stmt.line)
            return

        if isinstance(stmt, IfStmt):
            cond_val = self._truthy(self._eval_expr(stmt.cond, env))
            branch = "then" if cond_val else "else"
            self._trace("if_branch", line=stmt.line, branch=branch, condition=cond_val)
            body = stmt.then_body if cond_val else stmt.else_body
            self._exec_block(body, env)
            return

        if isinstance(stmt, ForStmt):
            iterable = self._eval_expr(stmt.iterable_expr, env)
            if not isinstance(iterable, list):
                self._runtime_error("for-loop iterable must be a list", node=stmt)
            for item in iterable:
                if stmt.var_name not in env.vars:
                    try:
                        env.define(stmt.var_name, item)
                    except PlanRuntimeError as exc:
                        self._runtime_error(str(exc), node=stmt)
                else:
                    try:
                        env.set(stmt.var_name, item)
                    except PlanRuntimeError as exc:
                        self._runtime_error(str(exc), node=stmt)
                self._trace("for_iteration", line=stmt.line, target=stmt.var_name, value=item)
                self._exec_block(stmt.body, env)
            return

        if isinstance(stmt, TryCatchStmt):
            try:
                self._exec_block(stmt.try_body, env)
            except ToolError as exc:
                if stmt.error_var not in env.vars:
                    try:
                        env.define(stmt.error_var, exc)
                    except PlanRuntimeError as err:
                        self._runtime_error(str(err), node=stmt)
                else:
                    try:
                        env.set(stmt.error_var, exc)
                    except PlanRuntimeError as err:
                        self._runtime_error(str(err), node=stmt)
                self._trace("catch_tool_error", line=stmt.line, error=str(exc))
                self._exec_block(stmt.catch_body, env)
            return

        if isinstance(stmt, ReturnStmt):
            value = self._eval_expr(stmt.expr, env) if stmt.expr is not None else None
            self._trace("return_stmt", line=stmt.line, value=value)
            raise ReturnSignal(value)

        if isinstance(stmt, ExprStmt):
            result = self._eval_expr(stmt.expr, env)
            self._trace("expr_stmt", line=stmt.line, value=result)
            return

        self._runtime_error(f"Unknown statement type: {stmt}", node=stmt)

    # ------------------------------------------------------------------
    # Expressions

    def _eval_expr(self, expr: Expr, env: Environment) -> Any:
        if isinstance(expr, Literal):
            return expr.value

        if isinstance(expr, VarRef):
            try:
                return env.get(expr.name)
            except PlanRuntimeError as exc:
                self._runtime_error(str(exc), node=expr)

        if isinstance(expr, ListLiteral):
            return [self._eval_expr(e, env) for e in expr.elements]

        if isinstance(expr, MapLiteral):
            return {k: self._eval_expr(v, env) for k, v in expr.items.items()}

        if isinstance(expr, CallExpr):
            arg_vals = [self._eval_expr(a, env) for a in expr.args]
            return self.call_function(expr.name, arg_vals, call_node=expr)

        if isinstance(expr, SyscallExpr):
            if not self.registry.has(expr.name):
                self._runtime_error(f"Syscall '{expr.name}' not registered", node=expr)
            fn = self.registry.get(expr.name)
            arg_vals = [self._eval_expr(a, env) for a in expr.args]
            self._trace("syscall_start", name=expr.name, args=arg_vals, line=expr.line)
            try:
                result = fn(*arg_vals)
                self._trace("syscall_end", name=expr.name, result=result, line=expr.line)
                return result
            except ToolError as err:
                self._trace("syscall_error", name=expr.name, error=str(err), line=expr.line)
                location = self._format_location(expr)
                suffix = f" at {location}" if location else ""
                raise ToolError(f"Syscall '{expr.name}' failed{suffix}: {err}") from err
            except Exception as exc:
                self._trace("syscall_error", name=expr.name, error=str(exc), line=expr.line)
                location = self._format_location(expr)
                suffix = f" at {location}" if location else ""
                raise ToolError(f"Syscall '{expr.name}' raised unexpected error{suffix}: {exc}") from exc

        if isinstance(expr, BinaryOp):
            left = self._eval_expr(expr.left, env)
            if expr.op == "&&":
                if not self._truthy(left):
                    return False
                right = self._eval_expr(expr.right, env)
                return self._truthy(left) and self._truthy(right)
            if expr.op == "||":
                if self._truthy(left):
                    return True
                right = self._eval_expr(expr.right, env)
                return self._truthy(left) or self._truthy(right)

            right = self._eval_expr(expr.right, env)
            if expr.op == "==":
                return left == right
            if expr.op == "!=":
                return left != right
            if expr.op == "+":
                try:
                    return left + right
                except TypeError as exc:
                    self._runtime_error(
                        f"Cannot apply '+' to {type(left).__name__} and {type(right).__name__}",
                        node=expr,
                    )
            self._runtime_error(f"Unknown binary operator {expr.op}", node=expr)

        if isinstance(expr, str):
            try:
                return env.get(expr)
            except PlanRuntimeError as exc:
                self._runtime_error(str(exc), node=None)

        self._runtime_error(f"Unknown expression type: {expr}")

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _truthy(value: Any) -> bool:
        return bool(value)

    def _validate_plan(self):
        validator = PlanValidator(available_syscalls=set(self.syscalls.keys()))
        validator.validate(self.plan)

    def _trace(self, event_type: str, **payload):
        if not hasattr(self, "tracer") or self.tracer is None:
            return
        self.tracer.emit(event_type, **payload)

    def _runtime_error(self, message: str, node: Optional[Any] = None):
        parts: List[str] = []
        current_function = self._call_stack[-1] if self._call_stack else None
        if current_function:
            parts.append(f"in function '{current_function}'")
        line = getattr(node, "line", None) if node is not None else None
        column = getattr(node, "column", None) if node is not None else None
        if line is not None:
            parts.append(f"line {line}")
        if column is not None:
            parts.append(f"column {column}")
        suffix = f" ({', '.join(parts)})" if parts else ""
        full_message = f"{message}{suffix}"
        self._trace(
            "error",
            message=full_message,
            context="runtime",
            function=current_function,
            line=line,
            column=column,
        )
        raise PlanRuntimeError(full_message)

    @staticmethod
    def _format_location(node: Optional[Any]) -> str:
        if node is None:
            return ""
        line = getattr(node, "line", None)
        column = getattr(node, "column", None)
        parts = []
        if line is not None:
            parts.append(f"line {line}")
        if column is not None:
            parts.append(f"column {column}")
        return ", ".join(parts)


__all__ = [
    "ExecutionTracer",
    "PlanInterpreter",
    "PlanRuntimeError",
]
