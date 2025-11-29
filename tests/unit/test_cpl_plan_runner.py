"""Unit tests for :mod:`llmflow.planning.plan_runner`."""
from __future__ import annotations

from typing import Callable, Dict, List

from dsl.syscall_registry import SyscallRegistry

from llmflow.planning.plan_runner import CPLPlanRunner


def _registry_factory(capture: List[str]) -> Callable[[], SyscallRegistry]:
    def factory() -> SyscallRegistry:
        registry = SyscallRegistry()

        def log(message: str) -> None:
            capture.append(message)
            return None

        registry.register("log", log)
        return registry

    return factory


def test_execute_runs_plan_and_merges_metadata():
    messages: List[str] = []
    runner = CPLPlanRunner(
        registry_factory=_registry_factory(messages),
        dsl_specification="SPEC",
    )
    plan_source = """plan {
        function main() : Void {
            syscall.log(\"hello\");
            return;
        }
    }
    """
    metadata = {"request_id": "abc"}

    result = runner.execute(plan_source, capture_trace=True, metadata=dict(metadata))

    assert result["success"] is True
    assert messages == ["hello"]
    assert result["metadata"]["has_trace"] is True
    assert result["metadata"]["functions"] == 1
    assert result["metadata"]["request_id"] == "abc"
    assert metadata == {"request_id": "abc"}
    assert isinstance(result["trace"], list)


def test_execute_invokes_deferred_planner_with_context():
    captured_prompt: Dict[str, object] = {}

    def deferred_planner(prompt):
        captured_prompt["prompt"] = prompt
        return '{ syscall.log("deferred"); return; }'

    messages: List[str] = []
    runner = CPLPlanRunner(
        registry_factory=_registry_factory(messages),
        deferred_planner=deferred_planner,
        dsl_specification="SPEC",
    )

    plan_source = """plan {
        @Deferred
        function deferredTask() : Void;

        function main() : Void {
            deferredTask();
            return;
        }
    }
    """

    result = runner.execute(
        plan_source,
        metadata={"request_id": "xyz"},
        goal_summary="demo",
        deferred_metadata={"tier": "hot"},
        deferred_constraints=["Rule"],
    )

    assert result["success"] is True
    assert messages == ["deferred"]
    prompt = captured_prompt["prompt"]
    assert prompt.context.goal_summary == "demo"
    assert prompt.context.extra_metadata["tier"] == "hot"
    assert prompt.context.call_stack == ["main"]
    assert prompt.allowed_syscalls == ["log"]
