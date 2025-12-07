from __future__ import annotations

from pathlib import Path
from typing import List

from llmflow.planning import (
    JavaCompilationResult,
    JavaPlanRequest,
    JavaPlanner,
    PlanOrchestrator,
)
from llmflow.planning.executor import PlanExecutor
from llmflow.runtime.syscall_registry import SyscallRegistry
_TOOL_STUB = """
@SuppressWarnings("unused")
public final class PlanningToolStubs {
    private PlanningToolStubs() {}

    public static void log(String message) {}
}
""".strip()



class SequenceLLMClient:
    """Deterministic LLM stub that yields predefined Java plans."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.provider = type("Provider", (), {"max_retries": 0})()

    def structured_generate(self, *, messages, response_model, **kwargs):
        if not self._responses:
            raise RuntimeError("No more responses queued for SequenceLLMClient")
        self.calls += 1
        payload = self._responses.pop(0)
        return response_model(**payload)


class AlwaysSuccessfulCompiler:
    def compile(self, plan_source: str, **kwargs) -> JavaCompilationResult:
        work_dir = kwargs.get("working_dir")
        if work_dir is not None:
            work_path = Path(work_dir)
            work_path.mkdir(parents=True, exist_ok=True)
            (work_path / "Plan.class").write_bytes(b"")
        return JavaCompilationResult(success=True, command=("javac",), stdout="", stderr="")

def _build_runner() -> PlanExecutor:
    registry = SyscallRegistry()

    def _log(message: str):
        del message
        return {"success": True}

    registry.register("log", _log)
    return PlanExecutor(registry)


def test_java_retry_loop_end_to_end():
    first_plan_missing_main = {
        "java": """
        public class Plan {
            public void helper() {
                PlanningToolStubs.log("noop");
                return;
            }
        }
        """,
    }
    second_plan_succeeds = {
        "java": """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("success");
                return;
            }
        }
        """,
    }

    llm = SequenceLLMClient([first_plan_missing_main, second_plan_succeeds])
    planner = JavaPlanner(llm, specification="SPEC", structured_enabled=True)
    orchestrator = PlanOrchestrator(
        planner,
        runner_factory=_build_runner,
        max_retries=1,
        plan_compiler=AlwaysSuccessfulCompiler(),
        enable_plan_cache=False,
    )

    request = JavaPlanRequest(
        task="Demonstrate retry",
        goals=["ship"],
        tool_names=["log"],
        tool_stub_class_name="PlanningToolStubs",
        tool_stub_source=_TOOL_STUB,
    )

    result = orchestrator.execute_with_retries(request, capture_trace=True)

    assert result["success"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["execution"]["success"] is False
    assert result["attempts"][0]["execution"]["errors"][0]["type"] == "validation_error"
    assert result["telemetry"]["attempt_count"] == 2
    assert "Attempt 1" in result["summary"]
    assert llm.calls == 2