"""Unit tests for :mod:`llmflow.planning.plan_runner`."""
from __future__ import annotations

from llmflow.planning.plan_runner import PlanRunner


def test_execute_returns_graph_metadata():
    runner = PlanRunner(specification="SPEC")
    plan_source = """
    public class Plan {
        public void helper(String message) {
            PlanningToolStubs.log(message);
            return;
        }

        public void main() {
            helper("hi");
        }
    }
    """

    result = runner.execute(
        plan_source,
        metadata={"request_id": "abc"},
        tool_stub_class_name="PlanningToolStubs",
    )

    assert result["success"] is True
    assert result["errors"] == []
    assert isinstance(result["graph"], dict)
    assert result["metadata"]["functions"] == 2
    assert result["metadata"]["function_names"] == ["helper", "main"]
    assert result["metadata"]["tool_call_count"] == 1
    assert result["metadata"]["request_id"] == "abc"


def test_execute_reports_missing_main():
    runner = PlanRunner(specification="SPEC")
    plan_source = """
    public class Plan {
        public void helper() {
            PlanningToolStubs.log("noop");
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
            PlanningToolStubs.log("oops");
        }
    }
    """

    result = runner.execute(plan_source)

    assert result["success"] is False
    assert result["errors"][0]["type"] == "parse_error"


def test_execute_reports_lexer_error_with_markdown_header():
    runner = PlanRunner(specification="SPEC")
    plan_source = """# Java Planning Specification\npublic class Plan {\n    public void main() {\n        PlanningToolStubs.log(\"noop\");\n    }\n}\n"""

    result = runner.execute(plan_source)

    assert result["success"] is False
    assert result["errors"][0]["type"] == "parse_error"
    assert "Java Planning Specification" in result["errors"][0]["message"]
