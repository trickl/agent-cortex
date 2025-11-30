"""Utilities for traversing and decomposing Java plans into actionable sub-goals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .java_plan_analysis import (
    HelperInvocation,
    JavaPlanGraph,
    FunctionSummary,
    ToolInvocation,
    analyze_java_plan,
)


@dataclass
class PlanSubgoalIntent:
    """Represents the next actionable unit extracted from a Java plan."""

    goal: str
    action_kind: str
    action_name: str
    parent_function: str
    comment: Optional[str]
    line: Optional[int]
    column: Optional[int]
    args: List[str]


class JavaPlanNavigator:
    """Interprets Java plan graphs to produce sub-goal intents."""

    def __init__(self, graph: JavaPlanGraph) -> None:
        self._graph = graph
        self._functions: Dict[str, FunctionSummary] = {fn.name: fn for fn in graph.functions}

    @classmethod
    def from_source(
        cls,
        plan_source: str,
        *,
        tool_stub_class_name: Optional[str] = None,
    ) -> "JavaPlanNavigator":
        graph = analyze_java_plan(plan_source, tool_stub_class_name=tool_stub_class_name)
        return cls(graph)

    def next_subgoal(self) -> Optional[PlanSubgoalIntent]:
        entry = self._select_entry_function()
        if entry is None:
            return None

        target_function = entry
        if self._should_inline(entry):
            inline_call = entry.helper_calls[0]
            helper_fn = self._functions.get(inline_call.name)
            if helper_fn is None:
                return self._intent_from_invocation("helper", inline_call, entry.name)
            target_function = helper_fn

        invocation = self._pick_first_invocation(target_function)
        if invocation is None:
            return None
        kind, call = invocation
        return self._intent_from_invocation(kind, call, target_function.name)

    # ------------------------------------------------------------------

    def _select_entry_function(self) -> Optional[FunctionSummary]:
        for function in self._graph.functions:
            if function.name == "main":
                return function
        return self._graph.functions[0] if self._graph.functions else None

    @staticmethod
    def _should_inline(function: FunctionSummary) -> bool:
        return len(function.helper_calls) == 1 and not function.tool_calls

    @staticmethod
    def _pick_first_invocation(
        function: FunctionSummary,
    ) -> Optional[Tuple[str, HelperInvocation | ToolInvocation]]:
        candidates: List[Tuple[str, HelperInvocation | ToolInvocation]] = []
        for call in function.helper_calls:
            candidates.append(("helper", call))
        for call in function.tool_calls:
            candidates.append(("tool", call))
        if not candidates:
            return None

        def _key(item: Tuple[str, HelperInvocation | ToolInvocation]) -> Tuple[int, int]:
            _, call = item
            line = call.line if call.line is not None else 10**9
            column = call.column if call.column is not None else 10**9
            return (line, column)

        candidates.sort(key=_key)
        return candidates[0]

    def _intent_from_invocation(
        self,
        kind: str,
        call: HelperInvocation | ToolInvocation,
        parent_function: str,
    ) -> PlanSubgoalIntent:
        summary = call.comment or self._default_goal(kind, call.name)
        return PlanSubgoalIntent(
            goal=summary,
            action_kind=kind,
            action_name=call.name,
            parent_function=parent_function,
            comment=call.comment,
            line=call.line,
            column=call.column,
            args=list(call.args),
        )

    @staticmethod
    def _default_goal(kind: str, name: str) -> str:
        if kind == "tool":
            return f"Call planning tool {name}"
        return f"Execute helper {name}"


__all__ = ["JavaPlanNavigator", "PlanSubgoalIntent"]
