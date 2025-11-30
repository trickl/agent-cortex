"""Regression tests for Java plan analysis expression rendering."""
from __future__ import annotations

from llmflow.planning.java_plan_analysis import analyze_java_plan


def _find_function(graph, name: str):
    for fn in graph.functions:
        if fn.name == name:
            return fn
    raise AssertionError(f"Function {name} not found")


def test_analyzer_renders_complex_assignments():
    source = """
    public class Plan {
        public int compute(int left, int right) {
            int sum = (left + right);
            int[] results = new int[2];
            results[0] = sum;
            PlanningToolStubs.log("sum=" + sum);
            return sum;
        }

        public void main() {
            int value = compute(1, 2);
        }
    }
    """

    graph = analyze_java_plan(source, tool_stub_class_name="PlanningToolStubs")
    compute_fn = _find_function(graph, "compute")
    assignments = [assignment.expression for assignment in compute_fn.assignments]

    assert "(left + right)" in assignments[0]
    assert assignments[1] == "ArrayCreator"
    assert assignments[2] == "sum"
    assert compute_fn.tool_calls[0].name == "log"


def test_graph_dict_contains_helper_metadata():
    source = """
    public class Plan {
        public void helper() {
            PlanningToolStubs.log("hi");
        }

        public void main() {
            helper();
        }
    }
    """

    graph_dict = analyze_java_plan(source, tool_stub_class_name="PlanningToolStubs").to_dict()
    functions = {fn["name"]: fn for fn in graph_dict["functions"]}

    assert "helper" in functions
    assert functions["helper"]["tool_calls"][0]["name"] == "log"
    main_calls = [call["name"] for call in functions["main"]["helper_calls"]]
    assert main_calls == ["helper"]