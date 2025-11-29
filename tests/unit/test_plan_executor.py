import pytest

from llmflow.planning.executor import PlanExecutor


def test_plan_executor_stub_raises_runtime_error():
    with pytest.raises(RuntimeError, match="PlanExecutor has been removed"):
        PlanExecutor()
