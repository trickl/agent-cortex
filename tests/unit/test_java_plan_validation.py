import pytest

from llmflow.planning.java_plan_analysis import analyze_java_plan
from llmflow.planning.plan_runner import PlanRunner


def test_analyze_java_plan_requires_class_declaration():
    source = "public interface Empty {}"
    with pytest.raises(ValueError):
        analyze_java_plan(source)


def test_analyze_java_plan_prefers_plan_named_class():
    source = """
    public class Helper {
        public void noop() {}
    }

    public class Plan {
        public void main() {}
    }
    """

    graph = analyze_java_plan(source)
    assert graph.class_name == "Plan"
    assert graph.functions[0].name == "main"


def test_plan_runner_reports_validation_errors():
    runner = PlanRunner(specification="SPEC")
    source = """
    public class Plan {
        public void helper() {}
    }
    """

    result = runner.execute(source)

    assert result["success"] is False
    error_types = {error["type"] for error in result["errors"]}
    assert "validation_error" in error_types


def test_plan_runner_flags_stub_helpers():
    runner = PlanRunner(specification="SPEC")
    source = """
    public class Plan {
        public void main() {
            // orchestration entrypoint
        }

        private boolean hasOpenIssues() {
            // Stub: Check if there are any open issues in Qlty.
            return false;
        }
    }
    """

    result = runner.execute(source)

    assert result["success"] is False
    assert any(error["type"] == "stub_method" for error in result["errors"])
