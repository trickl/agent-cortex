import pytest

from llmflow.planning import CPLPlanRequest, CPLPlanner, CPLPlanningError


class DummyLLMClient:
    def __init__(self, content: str):
        self._content = content
        self.messages = None

    def generate(self, messages, tools=None):
        self.messages = messages
        return {"role": "assistant", "content": self._content}


def test_planner_builds_prompt_and_returns_plan():
    client = DummyLLMClient(
        """plan {
            function main() : Void {
                syscall.log(\"hello\");
                return;
            }
        }
        """
    )
    planner = CPLPlanner(client, dsl_specification="SPEC CONTENT")
    request = CPLPlanRequest(
        task="Fix the reported lint issue",
        goals=["Diagnose the lint failure", "Apply a minimal patch"],
        context="Repository: example, Branch: main",
        allowed_syscalls=["log", "cloneRepo"],
    )

    result = planner.generate_plan(request)

    assert result.plan_source.lstrip().startswith("plan")
    assert result.metadata["allowed_syscalls"] == ["cloneRepo", "log"]
    assert client.messages[0]["role"] == "system"
    assert "SPEC CONTENT" in client.messages[0]["content"]
    assert "cloneRepo" in client.messages[1]["content"]


def test_planner_rejects_invalid_response():
    client = DummyLLMClient("This is not a plan")
    planner = CPLPlanner(client, dsl_specification="SPEC")

    with pytest.raises(CPLPlanningError):
        planner.generate_plan(CPLPlanRequest(task="Summarize"))
