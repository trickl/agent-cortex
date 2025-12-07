"""Static analysis helpers for Java planner output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import javalang


@dataclass
class ToolInvocation:
    """Represents a direct planning tool invocation discovered in the plan."""

    name: str
    args: List[str]
    line: Optional[int]
    column: Optional[int]
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": list(self.args),
            "line": self.line,
            "column": self.column,
            "comment": self.comment,
        }


@dataclass
class HelperInvocation:
    """Invocation of another helper method within the plan."""

    name: str
    args: List[str]
    line: Optional[int]
    column: Optional[int]
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": list(self.args),
            "line": self.line,
            "column": self.column,
            "comment": self.comment,
        }


@dataclass
class StateAssignment:
    """Represents local state captured via declarations or assignments."""

    target: str
    expression: str
    line: Optional[int]
    column: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "expression": self.expression,
            "line": self.line,
            "column": self.column,
        }


@dataclass
class BranchSummary:
    """Summary of branching logic (if/else or ternary expressions)."""

    kind: str
    condition: str
    line: Optional[int]
    column: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "condition": self.condition,
            "line": self.line,
            "column": self.column,
        }


@dataclass
class ExceptionHandlerSummary:
    """Captures try/catch usage so orchestrators can reason about fallbacks."""

    error_var: str
    line: Optional[int]
    column: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_var": self.error_var,
            "line": self.line,
            "column": self.column,
        }


@dataclass
class FunctionSummary:
    """Aggregated metadata for a single helper function."""

    name: str
    params: List[str]
    return_type: Optional[str]
    tool_calls: List[ToolInvocation] = field(default_factory=list)
    helper_calls: List[HelperInvocation] = field(default_factory=list)
    assignments: List[StateAssignment] = field(default_factory=list)
    branches: List[BranchSummary] = field(default_factory=list)
    exception_handlers: List[ExceptionHandlerSummary] = field(default_factory=list)
    is_stub: bool = False
    stub_comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": list(self.params),
            "return_type": self.return_type,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "helper_calls": [call.to_dict() for call in self.helper_calls],
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "branches": [branch.to_dict() for branch in self.branches],
            "exception_handlers": [handler.to_dict() for handler in self.exception_handlers],
            "is_stub": self.is_stub,
            "stub_comment": self.stub_comment,
        }


@dataclass
class JavaPlanGraph:
    """High-level representation of the Java plan suitable for orchestration."""

    class_name: str
    functions: List[FunctionSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "functions": [fn.to_dict() for fn in self.functions],
        }


def analyze_java_plan(
    source: str,
    tool_stub_class_name: Optional[str] = None,
) -> JavaPlanGraph:
    """Parse ``source`` and extract helper/tool relationships."""

    tree = javalang.parse.parse(source)
    class_decl = _select_plan_class(tree.types)
    source_lines = source.splitlines()
    functions: List[FunctionSummary] = []
    qualifiers = {"syscall"}
    if tool_stub_class_name:
        qualifiers.add(tool_stub_class_name)
    for method in class_decl.methods:
        functions.append(_summarize_method(method, source_lines, qualifiers))
    return JavaPlanGraph(class_name=class_decl.name, functions=functions)


def _select_plan_class(types: Sequence[Any]) -> javalang.tree.ClassDeclaration:
    classes = [node for node in types if isinstance(node, javalang.tree.ClassDeclaration)]
    if not classes:
        raise ValueError("Java plan must declare at least one class")
    for cls in classes:
        if cls.name == "Plan" or cls.name == "Planner":
            return cls
    return classes[0]


def _summarize_method(
    method: javalang.tree.MethodDeclaration,
    source_lines: Sequence[str],
    tool_qualifiers: Sequence[str],
) -> FunctionSummary:
    params = [param.name for param in method.parameters]
    summary = FunctionSummary(
        name=method.name,
        params=params,
        return_type=_type_name(method.return_type),
    )
    if method.body is None:
        summary.is_stub = method.name != "main"
        summary.stub_comment = None
        return summary

    normalized_qualifiers = {qualifier.split(".")[-1] for qualifier in tool_qualifiers}
    for _, node in method.filter(javalang.tree.MethodInvocation):
        args = [_render_expression(arg) for arg in (node.arguments or [])]
        line, column = _node_position(node)
        comment = _extract_leading_comment(source_lines, line)
        qualifier = (node.qualifier or "").split(".")[-1]
        if qualifier in normalized_qualifiers:
            summary.tool_calls.append(
                ToolInvocation(
                    name=node.member,
                    args=args,
                    line=line,
                    column=column,
                    comment=comment,
                )
            )
            continue
        summary.helper_calls.append(
            HelperInvocation(
                name=node.member,
                args=args,
                line=line,
                column=column,
                comment=comment,
            )
        )

    for _, node in method.filter(javalang.tree.LocalVariableDeclaration):
        position = _node_position(node)
        for declarator in node.declarators:
            summary.assignments.append(
                StateAssignment(
                    target=declarator.name,
                    expression=_render_expression(declarator.initializer),
                    line=position[0],
                    column=position[1],
                )
            )

    for _, node in method.filter(javalang.tree.Assignment):
        position = _node_position(node)
        summary.assignments.append(
            StateAssignment(
                target=_render_expression(node.expressionl),
                expression=_render_expression(node.value),
                line=position[0],
                column=position[1],
            )
        )

    for _, node in method.filter(javalang.tree.IfStatement):
        position = _node_position(node)
        summary.branches.append(
            BranchSummary(
                kind="if",
                condition=_render_expression(node.condition),
                line=position[0],
                column=position[1],
            )
        )

    for _, node in method.filter(javalang.tree.TernaryExpression):
        position = _node_position(node)
        summary.branches.append(
            BranchSummary(
                kind="ternary",
                condition=_render_expression(node),
                line=position[0],
                column=position[1],
            )
        )

    for _, node in method.filter(javalang.tree.TryStatement):
        for catch in node.catches:
            position = _node_position(catch)
            summary.exception_handlers.append(
                ExceptionHandlerSummary(
                    error_var=catch.parameter.name,
                    line=position[0],
                    column=position[1],
                )
            )

    summary.is_stub = summary.name != "main" and not summary.tool_calls and not summary.helper_calls
    if summary.is_stub:
        first_statement_line = _find_first_statement_line(method)
        if first_statement_line is not None:
            summary.stub_comment = _extract_leading_comment(source_lines, first_statement_line)
        else:
            summary.stub_comment = None

    return summary


def _find_first_statement_line(method: javalang.tree.MethodDeclaration) -> Optional[int]:
    if not method.body:
        return None
    queue: List[Any] = list(method.body)
    while queue:
        node = queue.pop(0)
        line, _ = _node_position(node)
        if line is not None:
            return line
        nested = getattr(node, "statements", None)
        if nested:
            queue[:0] = list(nested)
    return None


def _render_expression(node: Optional[Any]) -> str:
    if node is None:
        return "null"
    if isinstance(node, javalang.tree.Literal):
        return node.value
    if isinstance(node, javalang.tree.MemberReference):
        qualifier = f"{node.qualifier}." if node.qualifier else ""
        return f"{qualifier}{node.member}"
    if isinstance(node, javalang.tree.BinaryOperation):
        return f"({_render_expression(node.operandl)} {node.operator} {_render_expression(node.operandr)})"
    if isinstance(node, javalang.tree.MethodInvocation):
        qualifier = f"{node.qualifier}." if node.qualifier else ""
        args = ", ".join(_render_expression(arg) for arg in (node.arguments or []))
        return f"{qualifier}{node.member}({args})"
    if isinstance(node, javalang.tree.Cast):
        return f"(({_type_name(node.type)}) {_render_expression(node.expression)})"
    if isinstance(node, javalang.tree.TernaryExpression):
        return (
            f"({_render_expression(node.condition)} ? "
            f"{_render_expression(node.if_true)} : {_render_expression(node.if_false)})"
        )
    if isinstance(node, javalang.tree.This):
        return "this"
    if isinstance(node, javalang.tree.SuperMethodInvocation):
        args = ", ".join(_render_expression(arg) for arg in node.arguments)
        return f"super.{node.member}({args})"
    if isinstance(node, javalang.tree.ArraySelector):
        return f"{_render_expression(node.member)}[{_render_expression(node.index)}]"
    return node.__class__.__name__


def _type_name(type_node: Optional[Any]) -> Optional[str]:
    if type_node is None:
        return None
    if isinstance(type_node, javalang.tree.ReferenceType):
        base = ".".join(type_node.name) if isinstance(type_node.name, list) else type_node.name
        if type_node.arguments:
            rendered = ",".join(_type_name(arg.type) for arg in type_node.arguments if getattr(arg, "type", None))
            if rendered:
                base = f"{base}<{rendered}>"
        return base
    if hasattr(type_node, "name"):
        return str(type_node.name)
    return str(type_node)


def _node_position(node: Any) -> Tuple[Optional[int], Optional[int]]:
    position = getattr(node, "position", None)
    if not position:
        return None, None
    return position.line, position.column


def _extract_leading_comment(
    source_lines: Sequence[str],
    line_number: Optional[int],
) -> Optional[str]:
    if line_number is None or line_number <= 1:
        return None
    idx = line_number - 2  # zero-based index of line before invocation
    collected: List[str] = []
    encountered_content = False
    while idx >= 0:
        raw_line = source_lines[idx]
        stripped = raw_line.strip()
        if not stripped:
            if collected:
                break
            idx -= 1
            continue
        if stripped.startswith("//"):
            collected.append(stripped.lstrip("/").strip())
            idx -= 1
            continue
        if stripped.startswith("/*"):
            comment_line = stripped.lstrip("/*").rstrip("*/").strip()
            if comment_line:
                collected.append(comment_line)
            break
        encountered_content = True
        break
    if not collected:
        return None
    collected.reverse()
    return " ".join(part for part in collected if part)


__all__ = [
    "BranchSummary",
    "ExceptionHandlerSummary",
    "FunctionSummary",
    "JavaPlanGraph",
    "StateAssignment",
    "ToolInvocation",
    "HelperInvocation",
    "analyze_java_plan",
]
