from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import ast as py_ast  # for parsing string literals

from lark import Lark, Transformer, v_args, Token, Tree

from .deferred_planner import (
    DeferredFunctionContext,
    DeferredFunctionPrompt,
    DeferredParameter,
    prepare_deferred_prompt,
)

try:
    from .syscall_registry import SyscallRegistry
except ImportError:  # pragma: no cover
    from syscall_registry import SyscallRegistry  # type: ignore


_GRAMMAR_PATH = Path(__file__).with_name("cpl.lark")
_DSL_SPEC_PATH = Path(__file__).with_name("planning.md")
try:
    _DEFAULT_DSL_SPEC = _DSL_SPEC_PATH.read_text(encoding="utf-8")
except OSError:  # pragma: no cover - fallback when file missing
    _DEFAULT_DSL_SPEC = ""


# ============================================================
# Exceptions
# ============================================================

class CPLRuntimeError(Exception):
    """Generic runtime error in CPL execution."""
    pass


class ToolError(Exception):
    """Exception type corresponding to DSL ToolError."""
    pass


class DeferredSynthesisError(Exception):
    """Raised when generating a deferred function body fails."""


class ReturnSignal(Exception):
    """Internal control-flow exception used to implement return."""
    def __init__(self, value: Any):
        self.value = value


# ============================================================
# AST Node Definitions
# ============================================================

@dataclass
class Plan:
    functions: Dict[str, "FunctionDef"]
    ordered_functions: List["FunctionDef"]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class DeferredExecutionOptions:
    reuse_cached_bodies: bool = True
    goal_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra_constraints: List[str] = field(default_factory=list)


@dataclass
class Annotation:
    name: str
    args: List[Any]
    line: Optional[int] = None
    column: Optional[int] = None

    def matches(self, target: str) -> bool:
        return self.name == target


@dataclass
class FunctionDef:
    name: str
    params: List["Param"]
    return_type: str
    body: Optional[List["Stmt"]]
    annotations: List[Annotation] = field(default_factory=list)
    line: Optional[int] = None
    column: Optional[int] = None

    def is_deferred(self) -> bool:
        return any(annotation.name == "Deferred" for annotation in self.annotations)


@dataclass
class Param:
    name: str
    type: str
    line: Optional[int] = None
    column: Optional[int] = None


# --- Statements ---

class Stmt:
    pass


@dataclass
class VarDecl(Stmt):
    name: str
    type: str
    expr: "Expr"
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class Assign(Stmt):
    name: str
    expr: "Expr"
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class IfStmt(Stmt):
    cond: "Expr"
    then_body: List[Stmt]
    else_body: List[Stmt]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class ForStmt(Stmt):
    var_name: str
    iterable_expr: "Expr"
    body: List[Stmt]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class TryCatchStmt(Stmt):
    try_body: List[Stmt]
    error_var: str
    catch_body: List[Stmt]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class ReturnStmt(Stmt):
    expr: Optional["Expr"]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class ExprStmt(Stmt):
    expr: "Expr"
    line: Optional[int] = None
    column: Optional[int] = None


# --- Expressions ---

class Expr:
    pass


@dataclass
class Literal(Expr):
    value: Any
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class VarRef(Expr):
    name: str
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class ListLiteral(Expr):
    elements: List[Expr]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class MapLiteral(Expr):
    items: Dict[str, Expr]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class CallExpr(Expr):
    name: str
    args: List[Expr]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class SyscallExpr(Expr):
    name: str
    args: List[Expr]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class BinaryOp(Expr):
    op: str  # "||", "&&", "==", "!=", "+"
    left: Expr
    right: Expr
    line: Optional[int] = None
    column: Optional[int] = None


# ============================================================
# AST Builder (Lark Transformer)
# ============================================================

@v_args(inline=True)
class ASTBuilder(Transformer):

    @staticmethod
    def _pos(meta_like):
        line = getattr(meta_like, "line", None)
        column = getattr(meta_like, "column", None)
        return {"line": line, "column": column}

    # --- Top-level ---

    @v_args(meta=True, inline=True)
    def plan(self, meta, *functions):
        fn_map = {fn.name: fn for fn in functions}
        return Plan(functions=fn_map, ordered_functions=list(functions), **self._pos(meta))

    @v_args(meta=True, inline=True)
    def function(self, meta, *parts):
        if not parts:
            raise CPLRuntimeError("Malformed function definition")

        remaining = list(parts)
        annotations: List[Annotation] = []
        if remaining and isinstance(remaining[0], list) and (
            not remaining[0] or isinstance(remaining[0][0], Annotation)
        ):
            annotations = remaining.pop(0)

        if not remaining:
            raise CPLRuntimeError("Function definition missing name")
        name_token = remaining.pop(0)
        name = str(name_token)

        params: List[Param] = []
        if remaining and isinstance(remaining[0], list) and (
            not remaining[0] or isinstance(remaining[0][0], Param)
        ):
            params = remaining.pop(0)

        if not remaining:
            raise CPLRuntimeError("Function definition missing return type")
        return_type = str(remaining.pop(0))

        body_nodes = remaining.pop(0) if remaining else None
        body: Optional[List[Stmt]]
        if body_nodes is None:
            body = None
        else:
            body = list(body_nodes)

        return FunctionDef(
            name=name,
            params=params,
            return_type=return_type,
            body=body,
            annotations=annotations,
            **self._pos(meta),
        )

    def parameters(self, *params):
        return list(params)

    @v_args(meta=True, inline=True)
    def parameter(self, meta, name, type_):
        return Param(name=str(name), type=str(type_), **self._pos(meta))

    # --- Types ---

    def SIMPLE_TYPE(self, token: Token):
        return str(token)

    def list_type(self, *parts):
        inner_type = parts[-1]
        return f"List<{inner_type}>"

    def map_type(self, *parts):
        inner_type = parts[-1]
        return f"Map<String,{inner_type}>"

    # --- Statements ---

    @v_args(meta=True, inline=True)
    def var_decl(self, meta, name, type_, expr):
        return VarDecl(name=str(name), type=str(type_), expr=expr, **self._pos(meta))

    @v_args(meta=True, inline=True)
    def assign(self, meta, name, expr):
        return Assign(name=str(name), expr=expr, **self._pos(meta))

    def if_stmt(self, _if, _lp, cond, _rp, _l1, *rest):
        # rest = then_body..., "}" "else" "{" else_body..., "}"
        # We'll reconstruct using Tree structure instead (simpler):
        raise NotImplementedError("if_stmt should be handled via non-inline mode")


    def for_stmt(self, _for, _lp, var_name, _in_kw, iterable_expr, _rp, _lb, *body_and_rb):
        raise NotImplementedError("for_stmt should be handled via non-inline mode")

    def try_stmt(self, *args):
        raise NotImplementedError("try_stmt should be handled via non-inline mode")

    @v_args(meta=True, inline=True)
    def return_stmt(self, meta, *parts):
        expr = parts[0] if parts else None
        return ReturnStmt(expr=expr, **self._pos(meta))

    @v_args(meta=True, inline=True)
    def expr_stmt(self, meta, expr):
        return ExprStmt(expr=expr, **self._pos(meta))

    # --- Expressions: logical and equality operators ---

    @v_args(meta=True, inline=True)
    def or_op(self, meta, left, _op, right):
        return BinaryOp(op="||", left=left, right=right, **self._pos(meta))

    @v_args(meta=True, inline=True)
    def and_op(self, meta, left, _op, right):
        return BinaryOp(op="&&", left=left, right=right, **self._pos(meta))

    @v_args(meta=True, inline=True)
    def eq_op(self, meta, left, _op, right):
        return BinaryOp(op="==", left=left, right=right, **self._pos(meta))

    @v_args(meta=True, inline=True)
    def ne_op(self, meta, left, _op, right):
        return BinaryOp(op="!=", left=left, right=right, **self._pos(meta))

    @v_args(meta=True, inline=True)
    def concat(self, meta, left, _op, right):
        return BinaryOp(op="+", left=left, right=right, **self._pos(meta))

    # --- Atoms and literals ---

    def INT(self, token: Token):
        return Literal(int(token), line=token.line, column=token.column)

    def BOOL(self, token: Token):
        return Literal(True if token == "true" else False, line=token.line, column=token.column)

    def STRING(self, token: Token):
        # Use Python's literal_eval to unescape
        return Literal(py_ast.literal_eval(str(token)), line=token.line, column=token.column)

    @v_args(meta=True, inline=True)
    def list_literal(self, meta, *parts):
        # parts = "[" elements? "]"
        # But transformer usually gives elements directly, not brackets.
        return ListLiteral(elements=list(parts), **self._pos(meta))

    @v_args(meta=True, inline=True)
    def map_literal(self, meta, *items):
        # items are map_item nodes
        m = {}
        for key, expr in items:
            m[key] = expr
        return MapLiteral(items=m, **self._pos(meta))

    def map_item(self, key_token, expr):
        # key_token is STRING token
        key = py_ast.literal_eval(str(key_token))
        return key, expr

    @v_args(meta=True, inline=True)
    def call(self, meta, name, *maybe_args):
        args = list(maybe_args[0]) if maybe_args else []
        return CallExpr(name=str(name), args=args, **self._pos(meta))

    @v_args(meta=True, inline=True)
    def syscall_call(self, meta, name, *maybe_args):
        args = list(maybe_args[0]) if maybe_args else []
        return SyscallExpr(name=str(name), args=args, **self._pos(meta))

    def args_call(self, *exprs):
        return list(exprs)

    def atom(self, value):
        if isinstance(value, Token):
            return VarRef(name=str(value), line=value.line, column=value.column)
        return value

    def annotation_list(self, *annotations):
        return list(annotations)

    @v_args(meta=True, inline=True)
    def annotation_decl(self, meta, name, args=None):
        arg_list = list(args) if args else []
        return Annotation(name=str(name), args=arg_list, **self._pos(meta))

    def annotation_args(self, *args):
        if not args:
            return []
        return list(args[0])

    def annotation_arg_list(self, *items):
        return list(items)

    def annotation_arg(self, token: Token):
        if token.type == "STRING":
            return py_ast.literal_eval(str(token))
        return str(token)

    def function_block(self, *stmts):
        return list(stmts)

    def function_none(self, _token=None):
        return None

    # We need non-inline handlers for some block structures; we’ll override via Tree

# ============================================================
# A Second Transformer for Block Structures
# (Because inline handling of complex blocks is awkward)
# ============================================================

class BlockAwareTransformer(ASTBuilder):
    """
    Extends ASTBuilder but handles if/for/try from Tree shapes instead of inline.
    """

    @staticmethod
    def _blockify(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @v_args(meta=True, inline=True)
    def if_stmt(self, meta, cond, then_body, else_body):
        # children: cond, then_block, else_block
        return IfStmt(
            cond=cond,
            then_body=self._blockify(then_body),
            else_body=self._blockify(else_body),
            **self._pos(meta),
        )

    @v_args(meta=True, inline=True)
    def for_stmt(self, meta, var_name, iterable_expr, body):
        return ForStmt(
            var_name=str(var_name),
            iterable_expr=iterable_expr,
            body=self._blockify(body),
            **self._pos(meta),
        )

    @v_args(meta=True, inline=True)
    def try_stmt(self, meta, try_body, error_var, catch_body):
        return TryCatchStmt(
            try_body=self._blockify(try_body),
            error_var=str(error_var),
            catch_body=self._blockify(catch_body),
            **self._pos(meta),
        )

    def stmt(self, children):
        # Lark often wraps stmt rules; just unwrap single child.
        if len(children) == 1:
            return children[0]
        return children


# ============================================================
# Runtime Environment and VM
# ============================================================

class Environment:
    def __init__(self, parent: Optional["Environment"] = None):
        self.parent = parent
        self.vars: Dict[str, Any] = {}

    def define(self, name: str, value: Any):
        if name in self.vars:
            raise CPLRuntimeError(f"Variable '{name}' already defined")
        self.vars[name] = value

    def set(self, name: str, value: Any):
        env = self._find_env(name)
        if env is None:
            raise CPLRuntimeError(f"Variable '{name}' not defined")
        env.vars[name] = value

    def get(self, name: str) -> Any:
        env = self._find_env(name)
        if env is None:
            raise CPLRuntimeError(f"Variable '{name}' not defined")
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


class CPLInterpreter:
    """
    Executes a CPL Plan AST with a given syscall registry.

    syscalls: Dict[str, Callable[..., Any]]
    """

    def __init__(
        self,
        plan: Plan,
        syscalls: Optional[Dict[str, Callable[..., Any]]] = None,
        *,
        registry: Optional[SyscallRegistry] = None,
        tracer: Optional["ExecutionTracer"] = None,
        deferred_planner: Optional[Callable[[DeferredFunctionPrompt], str]] = None,
        deferred_options: Optional[DeferredExecutionOptions] = None,
        dsl_specification: Optional[str] = None,
        body_parser: Optional[Lark] = None,
    ):
        self.plan = plan
        if registry is not None and syscalls is not None:
            raise ValueError("Provide either 'syscalls' or 'registry', not both")
        if registry is None and syscalls is None:
            raise ValueError("A syscall mapping or registry must be provided")
        self.registry = registry or SyscallRegistry.from_mapping(syscalls or {})
        # Backwards compatibility for callers expecting the raw dict
        self.syscalls = self.registry.to_dict()
        self.tracer = tracer or ExecutionTracer(enabled=False)
        self._call_stack: List[str] = []
        self._deferred_planner = deferred_planner
        self._deferred_options = deferred_options or DeferredExecutionOptions()
        self._dsl_specification = dsl_specification if dsl_specification is not None else _DEFAULT_DSL_SPEC
        self._deferred_body_parser = body_parser
        self._validate_plan()

    def run(self) -> Any:
        if "main" not in self.plan.functions:
            self._runtime_error("No main() function defined", node=self.plan)
        self._trace("execution_start", entry_point="main")
        result = self.call_function("main", [], call_node=self.plan.functions["main"])
        self._trace("execution_end", entry_point="main", return_value=result)
        return result

    # --- Function Calls ---

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
        # Bind parameters
        for param, value in zip(fn.params, args):
            try:
                env.define(param.name, value)
            except CPLRuntimeError as exc:
                self._runtime_error(str(exc), node=param)

        fn_body = self._resolve_function_body(fn, args, call_node or fn)
        self._call_stack.append(name)
        self._trace("function_enter", function=name, args=args, line=fn.line)
        result: Any = None
        error: Optional[str] = None
        try:
            self._exec_block(fn_body, env)
        except ReturnSignal as rs:
            result = rs.value
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            self._trace("function_exit", function=name, return_value=result, error=error, line=fn.line)
            self._call_stack.pop()

        # Void return
        return result

    # --- Statement Execution ---

    def _exec_block(self, stmts: List[Stmt], env: Environment):
        for stmt in stmts:
            self._exec_stmt(stmt, env)

    def _resolve_function_body(
        self,
        fn: FunctionDef,
        args: List[Any],
        error_node: Optional[Any],
    ) -> List[Stmt]:
        if fn.is_deferred():
            if self._deferred_planner is None:
                self._runtime_error(
                    f"Deferred function '{fn.name}' cannot run without a planner",
                    node=error_node,
                )
            generated_flag = getattr(fn, "_deferred_generated", False)
            needs_refresh = (
                fn.body is None
                or not generated_flag
                or not self._deferred_options.reuse_cached_bodies
            )
            if needs_refresh:
                self._trace("deferred_generate_start", function=fn.name)
                try:
                    new_body = self._synthesize_deferred_body(fn, args)
                except DeferredSynthesisError as exc:
                    self._trace("deferred_generate_error", function=fn.name, error=str(exc))
                    self._runtime_error(str(exc), node=error_node)
                fn.body = new_body
                setattr(fn, "_deferred_generated", True)
                self._trace("deferred_generate_end", function=fn.name)
            if fn.body is None:
                self._runtime_error(
                    f"Deferred function '{fn.name}' did not produce a body",
                    node=error_node,
                )
            return fn.body

        if fn.body is None:
            self._runtime_error(f"Function '{fn.name}' is missing a body", node=error_node)
        return fn.body

    def _synthesize_deferred_body(self, fn: FunctionDef, args: List[Any]) -> List[Stmt]:
        if self._deferred_planner is None:
            raise DeferredSynthesisError(
                f"Deferred function '{fn.name}' requires a planner callback"
            )
        context = self._build_deferred_context(fn, args)
        constraints = self._deferred_options.extra_constraints or None
        prompt = prepare_deferred_prompt(
            context=context,
            dsl_specification=self._dsl_specification,
            allowed_syscalls=sorted(self.syscalls.keys()),
            extra_constraints=constraints,
        )
        planner_output = self._deferred_planner(prompt)
        if not isinstance(planner_output, str):
            raise DeferredSynthesisError(
                f"Deferred planner must return a string body (got {type(planner_output).__name__})"
            )
        body_text = planner_output.strip()
        if not body_text:
            raise DeferredSynthesisError(
                f"Deferred planner returned an empty body for '{fn.name}'"
            )
        normalized = self._normalize_body_source(body_text)
        return self._parse_deferred_body(fn, normalized)

    def _build_deferred_context(self, fn: FunctionDef, args: List[Any]) -> DeferredFunctionContext:
        argument_values = {param.name: value for param, value in zip(fn.params, args)}
        metadata = dict(self._deferred_options.metadata)
        parameters = [DeferredParameter(name=p.name, type=p.type) for p in fn.params]
        return DeferredFunctionContext(
            function_name=fn.name,
            return_type=fn.return_type,
            parameters=parameters,
            argument_values=argument_values,
            call_stack=list(self._call_stack),
            goal_summary=self._deferred_options.goal_summary,
            extra_metadata=metadata,
        )

    def _parse_deferred_body(self, fn: FunctionDef, body_text: str) -> List[Stmt]:
        params_src = ", ".join(f"{param.name}: {param.type}" for param in fn.params)
        function_source = (
            "plan {\n"
            f"    function {fn.name}({params_src}) : {fn.return_type} {body_text}\n"
            "}\n"
        )
        parser = self._get_deferred_parser()
        try:
            temp_plan = parse_cpl(function_source, parser)
        except Exception as exc:  # pragma: no cover - lark provides detailed errors
            raise DeferredSynthesisError(
                f"Deferred planner produced invalid syntax for '{fn.name}': {exc}"
            ) from exc
        generated = temp_plan.functions.get(fn.name)
        if generated is None or generated.body is None:
            raise DeferredSynthesisError(
                f"Deferred planner output did not contain a body for '{fn.name}'"
            )
        return generated.body

    def _get_deferred_parser(self) -> Lark:
        if self._deferred_body_parser is None:
            self._deferred_body_parser = load_cpl_parser(str(_GRAMMAR_PATH))
        return self._deferred_body_parser

    @staticmethod
    def _normalize_body_source(body_text: str) -> str:
        stripped = body_text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        if not stripped.startswith("{"):
            stripped = "{\n" + stripped
        if not stripped.endswith("}"):
            stripped = stripped + "\n}"
        return stripped

    def _exec_stmt(self, stmt: Stmt, env: Environment):
        if isinstance(stmt, VarDecl):
            value = self._eval_expr(stmt.expr, env)
            try:
                env.define(stmt.name, value)
            except CPLRuntimeError as exc:
                self._runtime_error(str(exc), node=stmt)
            self._trace("var_decl", name=stmt.name, value=value, line=stmt.line)
            return

        elif isinstance(stmt, Assign):
            value = self._eval_expr(stmt.expr, env)
            try:
                env.set(stmt.name, value)
            except CPLRuntimeError as exc:
                self._runtime_error(str(exc), node=stmt)
            self._trace("assignment", name=stmt.name, value=value, line=stmt.line)
            return

        elif isinstance(stmt, IfStmt):
            cond_val = self._truthy(self._eval_expr(stmt.cond, env))
            if cond_val:
                self._trace("if_branch", line=stmt.line, branch="then", condition=cond_val)
                self._exec_block(stmt.then_body, env)
            else:
                self._trace("if_branch", line=stmt.line, branch="else", condition=cond_val)
                self._exec_block(stmt.else_body, env)
            return

        elif isinstance(stmt, ForStmt):
            iterable = self._eval_expr(stmt.iterable_expr, env)
            if not isinstance(iterable, list):
                self._runtime_error("for-loop iterable must be a list", node=stmt)
            for item in iterable:
                if stmt.var_name not in env.vars:
                    try:
                        env.define(stmt.var_name, item)
                    except CPLRuntimeError as exc:
                        self._runtime_error(str(exc), node=stmt)
                else:
                    try:
                        env.set(stmt.var_name, item)
                    except CPLRuntimeError as exc:
                        self._runtime_error(str(exc), node=stmt)
                self._trace("for_iteration", line=stmt.line, target=stmt.var_name, value=item)
                self._exec_block(stmt.body, env)
            return

        elif isinstance(stmt, TryCatchStmt):
            try:
                self._exec_block(stmt.try_body, env)
            except ToolError as e:
                if stmt.error_var not in env.vars:
                    try:
                        env.define(stmt.error_var, e)
                    except CPLRuntimeError as exc:
                        self._runtime_error(str(exc), node=stmt)
                else:
                    try:
                        env.set(stmt.error_var, e)
                    except CPLRuntimeError as exc:
                        self._runtime_error(str(exc), node=stmt)
                self._trace("catch_tool_error", line=stmt.line, error=str(e))
                self._exec_block(stmt.catch_body, env)
            return

        elif isinstance(stmt, ReturnStmt):
            value = self._eval_expr(stmt.expr, env) if stmt.expr is not None else None
            self._trace("return_stmt", line=stmt.line, value=value)
            raise ReturnSignal(value)

        elif isinstance(stmt, ExprStmt):
            result = self._eval_expr(stmt.expr, env)
            self._trace("expr_stmt", line=stmt.line, value=result)
            return

        else:
            self._runtime_error(f"Unknown statement type: {stmt}", node=stmt)

    # --- Expression Evaluation ---

    def _eval_expr(self, expr: Expr, env: Environment) -> Any:
        if isinstance(expr, Literal):
            return expr.value

        if isinstance(expr, VarRef):
            try:
                return env.get(expr.name)
            except CPLRuntimeError as exc:
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
            except ToolError as tool_err:
                self._trace("syscall_error", name=expr.name, error=str(tool_err), line=expr.line)
                location = self._format_location(expr)
                suffix = f" at {location}" if location else ""
                raise ToolError(f"Syscall '{expr.name}' failed{suffix}: {tool_err}") from tool_err
            except Exception as e:
                self._trace("syscall_error", name=expr.name, error=str(e), line=expr.line)
                location = self._format_location(expr)
                suffix = f" at {location}" if location else ""
                raise ToolError(f"Syscall '{expr.name}' raised unexpected error{suffix}: {e}") from e

        if isinstance(expr, BinaryOp):
            left = self._eval_expr(expr.left, env)
            # short-circuit for && and ||
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

        # Fallback: NAME that wasn’t turned into VarRef yet (depending on transform)
        if isinstance(expr, str):
            try:
                return env.get(expr)
            except CPLRuntimeError as exc:
                self._runtime_error(str(exc), node=None)

        self._runtime_error(f"Unknown expression type: {expr}")

    @staticmethod
    def _truthy(value: Any) -> bool:
        return bool(value)

    def _validate_plan(self):
        from .cpl_validator import PlanValidator

        available_syscalls = set(self.syscalls.keys())
        validator = PlanValidator(available_syscalls=available_syscalls)
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
        raise CPLRuntimeError(full_message)

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


# ============================================================
# Parser Helper
# ============================================================

def load_cpl_parser(grammar_path: str) -> Lark:
    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = f.read()
    return Lark(grammar, start="plan", parser="lalr", propagate_positions=True)


def parse_cpl(source: str, parser: Lark) -> Plan:
    tree: Tree = parser.parse(source)
    transformer = BlockAwareTransformer()
    plan = transformer.transform(tree)
    return plan


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Example minimal syscall registry
    def log_syscall(msg: str):
        print("[LOG]", msg)

    def write_output_syscall(msg: str):
        print("[OUTPUT]", msg)

    registry = SyscallRegistry()
    registry.register("log", log_syscall)
    registry.register("writeOutput", write_output_syscall)

    example_source = r'''
    plan {

        function main() : Void {
            let msg: String = "Hello, CPL!";
            syscall.log(msg);
            syscall.writeOutput(msg + " Bye.");
            return;
        }
    }
    '''

    parser = load_cpl_parser("cpl.lark")
    plan = parse_cpl(example_source, parser)
    vm = CPLInterpreter(plan, registry=registry)
    vm.run()
