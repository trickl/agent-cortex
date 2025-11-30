from __future__ import annotations

import shutil

import pytest

from llmflow.planning import JavaPlanCompiler


_MINIMAL_PLAN = """
public class Plan {
    public void main() {
        PlanningToolStubs.log("hi");
    }
}
""".strip()

_TOOL_STUB = """
@SuppressWarnings("unused")
public final class PlanningToolStubs {
    private PlanningToolStubs() {}

    public static void log(String message) {}
}
""".strip()


def _require_javac() -> None:
    if shutil.which("javac") is None:
        pytest.skip("javac is not available on PATH")


def test_compiler_succeeds_for_minimal_plan():
    _require_javac()
    compiler = JavaPlanCompiler()

    result = compiler.compile(
        _MINIMAL_PLAN,
        tool_stub_source=_TOOL_STUB,
        tool_stub_class_name="PlanningToolStubs",
    )

    assert result.success is True
    assert result.errors == []
