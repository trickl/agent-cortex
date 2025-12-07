from __future__ import annotations

from llmflow.planning.executor import PlanExecutor
from llmflow.runtime.errors import ToolError
from llmflow.runtime.syscall_registry import SyscallRegistry


def _build_registry(log_impl):
    registry = SyscallRegistry()
    registry.register("log", log_impl)
    return registry


def test_plan_executor_executes_stubbed_plan_with_trace():
    calls = []

    def log_tool(message: str):
        calls.append(message)
        return {"success": True}

    executor = PlanExecutor(_build_registry(log_tool))
    plan_source = """
    public class Plan {
        public void helper() {
            PlanningToolStubs.log("hello");
        }

        public void main() {
            helper();
        }
    }
    """

    result = executor.execute_from_string(
        plan_source,
        capture_trace=True,
        metadata={"request_id": "abc"},
        tool_stub_class_name="PlanningToolStubs",
    )

    assert result["success"] is True
    assert result["errors"] == []
    assert result["return_value"] is None
    assert result["metadata"]["request_id"] == "abc"
    assert result["metadata"]["functions"] == 2
    assert result["metadata"]["tool_call_count"] == 1
    assert any(event["type"] == "syscall_start" for event in result["trace"])
    assert calls == ["hello"]


def test_plan_executor_reports_validation_failure():
    executor = PlanExecutor(_build_registry(lambda _: None))
    plan_source = """
    public class Plan {
        public void helper() {}
    }
    """

    result = executor.execute_from_string(plan_source, tool_stub_class_name="PlanningToolStubs")

    assert result["success"] is False
    assert result["errors"]
    assert result["errors"][0]["type"] == "validation_error"


def test_plan_executor_reports_parse_error():
    executor = PlanExecutor(_build_registry(lambda _: None))
    plan_source = """
    public class Plan {
        public void main( {
            PlanningToolStubs.log("oops");
        }
    }
    """

    result = executor.execute_from_string(plan_source, tool_stub_class_name="PlanningToolStubs")

    assert result["success"] is False
    assert result["errors"][0]["type"] == "parse_error"


def test_plan_executor_surfaces_tool_errors():
    def failing_tool(*_args):
        raise ToolError("log failed")

    executor = PlanExecutor(_build_registry(failing_tool))
    plan_source = """
    public class Plan {
        public void main() {
            PlanningToolStubs.log("boom");
        }
    }
    """

    result = executor.execute_from_string(plan_source, tool_stub_class_name="PlanningToolStubs")

    assert result["success"] is False
    assert result["errors"][0]["type"] == "tool_error"
    assert "log failed" in result["errors"][0]["message"]
