"""AST primitives for Java-based plans."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Plan:
    functions: Dict[str, "FunctionDef"]
    ordered_functions: List["FunctionDef"]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class FunctionDef:
    name: str
    params: List["Param"]
    return_type: str
    body: Optional[List["Stmt"]]
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class Param:
    name: str
    type: str
    line: Optional[int] = None
    column: Optional[int] = None


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
    op: str
    left: Expr
    right: Expr
    line: Optional[int] = None
    column: Optional[int] = None


__all__ = [
    "Assign",
    "BinaryOp",
    "CallExpr",
    "Expr",
    "ExprStmt",
    "ForStmt",
    "FunctionDef",
    "IfStmt",
    "ListLiteral",
    "Literal",
    "MapLiteral",
    "Param",
    "Plan",
    "ReturnStmt",
    "Stmt",
    "SyscallExpr",
    "TryCatchStmt",
    "VarDecl",
    "VarRef",
]
