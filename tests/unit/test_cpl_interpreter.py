"""Tests for the Java plan analysis graph (former interpreter suite)."""
from __future__ import annotations

from llmflow.planning.java_plan_analysis import analyze_java_plan


def _get_function(graph, name: str):
    for function in graph.functions:
        if function.name == name:
            return function
    raise AssertionError(f"Function {name} not found in graph")


def test_analyze_plan_captures_syscalls_helpers_and_state():
    source = """
    public class Plan {
        public void helper(String input) {
            String message = formatter.format(input);
            syscall.log(message);
            record(message);
            message = formatter.format(message);
        }

        public void record(String text) {
            syscall.log(text);
        }

        public void main() {
            String user = "world";
            helper(user);
            record(user);
        }
    }
    """

    graph = analyze_java_plan(source)
    helper = _get_function(graph, "helper")

    assert [call.name for call in helper.syscalls] == ["log"]
    helper_call_names = sorted(call.name for call in helper.helper_calls)
    assert helper_call_names == ["format", "format", "record"]
    assignments = helper.assignments
    assert assignments[0].target == "message"
    assert "formatter.format(input)" in assignments[0].expression
    assert assignments[1].expression.endswith("message)")


def test_analyze_plan_captures_branches_and_exception_handlers():
    source = """
    public class Plan {
        public boolean shouldExecute() {
            return true;
        }

        public void main() {
            try {
                if (shouldExecute()) {
                    syscall.log("run");
                } else {
                    syscall.log("skip");
                }
                String label = shouldExecute() ? "yes" : "no";
            } catch (Exception ex) {
                syscall.log(ex.getMessage());
            }
        }
    }
    """

    graph = analyze_java_plan(source)
    main_fn = _get_function(graph, "main")

    assert any(branch.kind == "if" for branch in main_fn.branches)
    assert any(branch.kind == "ternary" for branch in main_fn.branches)
    assert main_fn.exception_handlers[0].error_var == "ex"
    assert len(main_fn.syscalls) == 3
    assert main_fn.assignments[0].expression.startswith("(shouldExecute() ?")
