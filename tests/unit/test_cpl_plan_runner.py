"""Unit tests for :mod:`llmflow.planning.plan_runner`."""
from __future__ import annotations

from llmflow.planning.plan_runner import PlanRunner


def test_execute_returns_graph_metadata():
    runner = PlanRunner(specification="SPEC")
    plan_source = """
    public class Plan {
        public void helper(String message) {
            syscall.log(message);
            return;
        }

        public void main() {
            helper("hi");
        }
    }
    """

    result = runner.execute(plan_source, metadata={"request_id": "abc"})

    assert result["success"] is True
    assert result["errors"] == []
    assert isinstance(result["graph"], dict)
    assert result["metadata"]["functions"] == 2
    assert result["metadata"]["function_names"] == ["helper", "main"]
    assert result["metadata"]["syscall_count"] == 1
    assert result["metadata"]["request_id"] == "abc"


def test_execute_reports_missing_main():
    runner = PlanRunner(specification="SPEC")
    plan_source = """
    public class Plan {
        public void helper() {
            syscall.log("noop");
            return;
        }
    }
    """

    result = runner.execute(plan_source)

    assert result["success"] is False
    assert result["errors"][0]["type"] == "validation_error"


def test_execute_reports_parse_error():
    runner = PlanRunner(specification="SPEC")
    plan_source = """
    public class Plan {
        public void main( {
            syscall.log("oops");
        }
    }
    """

    result = runner.execute(plan_source)

    assert result["success"] is False
    assert result["errors"][0]["type"] == "parse_error"
