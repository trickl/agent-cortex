from dsl.plan_executor import PlanExecutor
from dsl.syscall_registry import SyscallRegistry


def make_executor(extra_syscalls=None):
    registry = SyscallRegistry()
    registry.register("log", lambda msg: msg)
    if extra_syscalls:
        for name, fn in extra_syscalls.items():
            registry.register(name, fn)
    return PlanExecutor(registry)


def test_plan_executor_happy_path_with_trace():
    executor = make_executor()
    plan = """plan {
        function main() : Void {
            syscall.log("hello");
            return;
        }
    }
    """
    result = executor.execute_from_string(plan, capture_trace=True, metadata={"request_id": "abc"})

    assert result["success"] is True
    assert result["return_value"] is None
    assert result["trace"], "Trace should be returned when capture_trace is True"
    assert result["metadata"]["request_id"] == "abc"


def test_plan_executor_missing_syscall_reports_validation_error():
    executor = make_executor()
    plan = """plan {
        function main() : Void {
            syscall.unknown();
            return;
        }
    }
    """
    result = executor.execute_from_string(plan)

    assert result["success"] is False
    assert result["errors"][0]["type"] == "validation_error"


def test_plan_executor_type_mismatch_detected():
    executor = make_executor()
    plan = """plan {
        function main() : Void {
            return;
        }

        function compute() : Int {
            return "oops";
        }
    }
    """
    result = executor.execute_from_string(plan)

    assert result["success"] is False
    assert result["errors"][0]["type"] == "validation_error"


def test_plan_executor_wrong_arg_count():
    executor = make_executor()
    plan = """plan {
        function main() : Void {
            helper();
            return;
        }

        function helper(val: Int) : Void {
            return;
        }
    }
    """
    result = executor.execute_from_string(plan)

    assert result["success"] is False
    assert result["errors"][0]["type"] == "validation_error"


def test_plan_executor_runtime_tool_error():
    def fail():
        raise RuntimeError("boom")

    executor = make_executor({"fail": fail})
    plan = """plan {
        function main() : Void {
            syscall.fail();
            return;
        }
    }
    """
    result = executor.execute_from_string(plan)
    assert result["success"] is False
    assert result["errors"][0]["type"] == "tool_error"
