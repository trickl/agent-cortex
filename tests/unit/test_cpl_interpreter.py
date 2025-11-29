from pathlib import Path
from typing import Callable, Dict, List

import pytest

from dsl.cpl_inerpreter import (
    CPLInterpreter,
    CPLRuntimeError,
    ExecutionTracer,
    SyscallRegistry,
    ToolError,
    load_cpl_parser,
    parse_cpl,
)
from dsl.cpl_validator import ValidationError


GRAMMAR_PATH = Path(__file__).resolve().parents[2] / "dsl" / "cpl.lark"


@pytest.fixture(scope="session")
def cpl_parser():
    return load_cpl_parser(str(GRAMMAR_PATH))


@pytest.fixture
def make_interpreter(cpl_parser):
    def _build(source: str, syscalls: Dict[str, Callable[..., object]], tracer: ExecutionTracer | None = None):
        plan = parse_cpl(source, cpl_parser)
        registry = SyscallRegistry.from_mapping(syscalls)
        return CPLInterpreter(plan, registry=registry, tracer=tracer)

    return _build


def test_interpreter_executes_plan(make_interpreter):
    messages: List[str] = []

    def log_syscall(msg: str):
        messages.append(msg)

    tracer = ExecutionTracer(enabled=True)
    interpreter = make_interpreter(
        """plan {

            function main() : Void {
                let items: List<String> = ["one", "two"];
                logItems(items);
                return;
            }

            function logItems(values: List<String>) : Void {
                for (value in values) {
                    syscall.log(value);
                }
                return;
            }
        }
        """,
        {"log": log_syscall},
        tracer=tracer,
    )

    interpreter.run()

    assert messages == ["one", "two"]
    assert any(event["type"] == "syscall_start" for event in tracer.as_list())


def test_missing_syscall_reports_line(make_interpreter):
    with pytest.raises(ValidationError) as exc_info:
        make_interpreter(
            """plan {
                function main() : Void {
                    syscall.unknown();
                    return;
                }
            }
            """,
            {},
        )

    assert "Syscall 'unknown' not registered" in str(exc_info.value)


def test_tool_error_caught_by_try_catch(make_interpreter):
    events: List[str] = []

    def flaky_syscall():
        raise ToolError("boom")

    def log_syscall(msg: str):
        events.append(msg)

    interpreter = make_interpreter(
        """plan {
            function main() : Void {
                try {
                    syscall.flaky();
                } catch (ToolError err) {
                    syscall.log("caught");
                }
                return;
            }
        }
        """,
        {"flaky": flaky_syscall, "log": log_syscall},
    )

    interpreter.run()

    assert events == ["caught"]
