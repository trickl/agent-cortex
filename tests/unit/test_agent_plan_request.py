from __future__ import annotations

from unittest.mock import MagicMock, patch

from llmflow.core.agent import Agent


class _DummyLLMClient:
    def generate(self, *_, **__):
        return {"content": "noop"}


def _make_agent() -> Agent:
    llm_client = _DummyLLMClient()
    planner = MagicMock(name="planner")
    return Agent(llm_client=llm_client, planner=planner, verbose=False, enable_run_logging=False)


def test_plan_request_uses_planner_class_name():
    agent = _make_agent()
    request = agent._build_plan_request("Investigate CI failures")

    assert request.task.startswith("Create a Java class named Planner"), request.task
    assert "User request:" in request.task
    assert "Investigate CI failures" in request.task
    assert request.context is None


def test_plan_request_context_skips_system_prompt():
    agent = _make_agent()
    agent._last_run_summary = "Handled flaky integration tests."
    agent.memory.add_user_message("Please fix the logger again.")
    agent.memory.add_assistant_message("Working on it.")

    request = agent._build_plan_request("Add retries to API client")
    assert request.context, "Planner context should include summary and conversation."
    assert "Previous plan summary" in request.context
    assert "Handled flaky integration tests." in request.context
    assert "Recent conversation:" in request.context
    assert "Please fix the logger again." in request.context
    assert "Working on it." in request.context
    assert "System prompt" not in request.context


def test_plan_request_includes_tool_stubs():
    stub_source = "public final class PlanningToolStubs {}"
    with patch(
        "llmflow.core.agent.generate_tool_stub_class",
        return_value=stub_source,
    ) as stub_builder:
        agent = _make_agent()
        request = agent._build_plan_request("Investigate CI failures")

    stub_builder.assert_called()
    assert request.tool_stub_source == stub_source
    assert request.tool_stub_class_name == "PlanningToolStubs"
