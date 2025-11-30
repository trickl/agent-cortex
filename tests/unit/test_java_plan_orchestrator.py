from dataclasses import dataclass
from typing import Dict, List

import pytest

from llmflow.planning import (
    CompilationError,
    JavaPlanRequest,
    JavaCompilationResult,
    PlanOrchestrator,
)


@dataclass
class StubPlanResult:
    plan_id: str
    plan_source: str
    metadata: Dict[str, str]
    raw_response: Dict[str, str]
    prompt_messages: List[Dict[str, str]]


class DummyPlanner:
    def __init__(self, plans: List[str]):
        self._plans = list(plans)
        self.requests: List[JavaPlanRequest] = []

    def generate_plan(self, request: JavaPlanRequest) -> StubPlanResult:
        if not self._plans:
            raise RuntimeError("No more plans queued")
        self.requests.append(request)
        source = self._plans.pop(0)
        return StubPlanResult(
            plan_id=f"stub-{len(self.requests)}",
            plan_source=source,
            metadata={},
            raw_response={},
            prompt_messages=[],
        )


class DummyRunner:
    def __init__(self, outcomes: List[Dict[str, object]]):
        self._outcomes = list(outcomes)
        self.calls: List[Dict[str, object]] = []

    def execute(self, plan_source: str, **kwargs) -> Dict[str, object]:
        if not self._outcomes:
            raise RuntimeError("No more outcomes queued")
        self.calls.append({"plan_source": plan_source, **kwargs})
        return self._outcomes.pop(0)


@pytest.fixture
def runner_factory():
    def _factory(outcomes):
        runner = DummyRunner(outcomes)

        def make_runner():
            return runner

        return runner, make_runner

    return _factory


class DummyCompiler:
    def __init__(self, results: List[JavaCompilationResult]):
        self._results = list(results)
        self.calls: List[Dict[str, object]] = []

    def compile(self, plan_source: str, **kwargs) -> JavaCompilationResult:
        if not self._results:
            raise RuntimeError("No more compilation results queued")
        self.calls.append({"plan_source": plan_source, **kwargs})
        return self._results.pop(0)


def _compile_success() -> JavaCompilationResult:
    return JavaCompilationResult(success=True, command=("javac",), stdout="", stderr="")


def _compile_failure(message: str) -> JavaCompilationResult:
    return JavaCompilationResult(
        success=False,
        command=("javac",),
        stdout="",
        stderr=message,
        errors=[CompilationError(message=message, file="Plan.java", line=3, column=5)],
    )


def test_orchestrator_succeeds_without_retries(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                return;
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([_compile_success()])
    orchestrator = PlanOrchestrator(planner, make_runner, plan_compiler=compiler)
    request = JavaPlanRequest(task="Do thing", goals=["goal"])

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is True
    assert len(result["attempts"]) == 1
    assert runner.calls[0]["plan_source"].lstrip().startswith("public class Plan")
    assert planner.requests[0].task == "Do thing"
    assert "telemetry" in result
    assert "summary" in result
    assert "Attempt 1" in result["summary"]


def test_orchestrator_retries_and_returns_failure(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                return;
            }
        }
        """,
        """
        public class Plan {
            public void main() {
                return;
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": False, "errors": [{"type": "validation_error", "message": "missing main", "line": 3}]},
        {"success": False, "errors": []},
    ])
    compiler = DummyCompiler([_compile_success(), _compile_success()])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=1,
        plan_compiler=compiler,
    )
    request = JavaPlanRequest(task="Do thing", goals=["goal"], additional_constraints=["Stay safe"])

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is False
    assert len(result["attempts"]) == 2
    assert planner.requests[1].metadata["attempt_index"] == 2
    assert len(planner.requests[1].additional_constraints) >= 2
    assert result["telemetry"]["attempt_count"] == 2
    assert result["telemetry"]["attempt_summaries"][-1]["status"] == "failure"


def test_telemetry_includes_tool_usage(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                return;
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {
            "success": True,
            "errors": [],
            "metadata": {"functions": 2},
            "trace": [
                {"type": "syscall_start", "name": "log"},
                {"type": "syscall_start", "name": "cloneRepo"},
                {"type": "syscall_start", "name": "log"},
            ],
        }
    ])
    compiler = DummyCompiler([_compile_success()])
    orchestrator = PlanOrchestrator(planner, make_runner, plan_compiler=compiler)
    request = JavaPlanRequest(task="Do thing", goals=["goal"])

    result = orchestrator.execute_with_retries(request)

    telemetry = result["telemetry"]
    assert telemetry["tool_usage"]["log"] == 2
    assert telemetry["tool_usage"]["cloneRepo"] == 1
    assert "cloneRepo" in result["summary"]


def test_compile_failure_triggers_refinement(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("bad");
            }
        }
        """,
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("fixed");
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([
        _compile_failure("cannot find symbol"),
        _compile_success(),
    ])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=0,
        max_compile_refinements=2,
        plan_compiler=compiler,
    )
    request = JavaPlanRequest(task="Fix it", goals=["goal"], tool_names=["log"], tool_stub_class_name="PlanningToolStubs")

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is True
    assert len(planner.requests) == 2, "Planner should be invoked for refinement"
    refine_request = planner.requests[1]
    assert "cannot find symbol" in refine_request.compile_error_report
    assert refine_request.prior_plan_source is not None
    attempt = result["attempts"][0]
    assert len(attempt["compile_attempts"]) == 2
    assert attempt["compile_attempts"][0]["success"] is False


def test_compile_failure_aborts_when_limit_reached(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("bad");
            }
        }
        """,
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("still bad");
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([
        _compile_failure("missing tool stub"),
        _compile_failure("still missing"),
    ])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=0,
        max_compile_refinements=2,
        plan_compiler=compiler,
    )
    request = JavaPlanRequest(task="Fix", goals=["goal"], tool_names=["log"], tool_stub_class_name="PlanningToolStubs")

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is False
    assert runner.calls == []
    errors = result["attempts"][0]["execution"]["errors"]
    assert errors and errors[0]["type"] == "compile_error"
