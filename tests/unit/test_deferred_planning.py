from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from dsl.cpl_inerpreter import (
    CPLInterpreter,
    CPLRuntimeError,
    DeferredExecutionOptions,
    ExecutionTracer,
    parse_cpl,
    load_cpl_parser,
)
from dsl.deferred_planner import DeferredFunctionPrompt

GRAMMAR_PATH = Path(__file__).resolve().parents[2] / "dsl" / "cpl.lark"


@pytest.fixture(scope="session")
def cpl_parser():
    return load_cpl_parser(str(GRAMMAR_PATH))


def _parse(source: str, parser):
    return parse_cpl(source, parser)


def _build_interpreter(source: str, parser, planner=None, options=None, log_sink: List[str] | None = None):
    plan = _parse(source, parser)
    messages: List[str] = log_sink if log_sink is not None else []

    def log_syscall(msg: str):
        messages.append(msg)

    interpreter = CPLInterpreter(
        plan,
        syscalls={"log": log_syscall},
        deferred_planner=planner,
        deferred_options=options,
        tracer=ExecutionTracer(enabled=False),
    )
    return interpreter, messages


def test_parse_deferred_function_declaration(cpl_parser):
    plan = _parse(
        """plan {
            @Deferred
            function perform() : Void;

            function main() : Void {
                return;
            }
        }""",
        cpl_parser,
    )

    perform = plan.functions["perform"]
    assert perform.is_deferred() is True
    assert perform.body is None


def test_parse_deferred_with_stub_body(cpl_parser):
    plan = _parse(
        """plan {
            @Deferred
            function perform() : Void {
                syscall.log("placeholder");
                return;
            }

            function main() : Void {
                return;
            }
        }""",
        cpl_parser,
    )

    perform = plan.functions["perform"]
    assert perform.is_deferred() is True
    assert perform.body is not None
    assert len(perform.body) == 2


def test_deferred_execution_invokes_planner_once(cpl_parser):
    planner_calls: List[DeferredFunctionPrompt] = []

    def planner(prompt: DeferredFunctionPrompt) -> str:
        planner_calls.append(prompt)
        return "{ syscall.log(\"from deferred\"); return; }"

    interpreter, messages = _build_interpreter(
        """plan {
            function main() : Void {
                perform();
                return;
            }

            @Deferred
            function perform() : Void;
        }""",
        cpl_parser,
        planner=planner,
    )

    interpreter.run()

    assert messages == ["from deferred"]
    assert len(planner_calls) == 1
    assert planner_calls[0].context.function_name == "perform"


def test_deferred_execution_reuses_cached_body(cpl_parser):
    planner_calls: List[DeferredFunctionPrompt] = []

    def planner(prompt: DeferredFunctionPrompt) -> str:
        planner_calls.append(prompt)
        return "{ syscall.log(\"cached run\"); return; }"

    interpreter, messages = _build_interpreter(
        """plan {
            function main() : Void {
                perform();
                perform();
                return;
            }

            @Deferred
            function perform() : Void;
        }""",
        cpl_parser,
        planner=planner,
    )

    interpreter.run()

    assert messages == ["cached run", "cached run"]
    assert len(planner_calls) == 1


def test_deferred_execution_without_cache_regenerates(cpl_parser):
    planner_calls: List[DeferredFunctionPrompt] = []

    def planner(prompt: DeferredFunctionPrompt) -> str:
        planner_calls.append(prompt)
        return "{ syscall.log(\"fresh run\"); return; }"

    options = DeferredExecutionOptions(reuse_cached_bodies=False)
    interpreter, _ = _build_interpreter(
        """plan {
            function main() : Void {
                perform();
                perform();
                return;
            }

            @Deferred
            function perform() : Void;
        }""",
        cpl_parser,
        planner=planner,
        options=options,
    )

    interpreter.run()

    assert len(planner_calls) == 2


def test_invalid_synthesis_raises_runtime_error(cpl_parser):
    def planner(_prompt: DeferredFunctionPrompt) -> str:
        return "not a block"

    interpreter, _ = _build_interpreter(
        """plan {
            function main() : Void {
                perform();
                return;
            }

            @Deferred
            function perform() : Void;
        }""",
        cpl_parser,
        planner=planner,
    )

    with pytest.raises(CPLRuntimeError):
        interpreter.run()


def test_nested_deferred_functions(cpl_parser):
    planner_calls: List[str] = []

    def planner(prompt: DeferredFunctionPrompt) -> str:
        planner_calls.append(prompt.context.function_name)
        if prompt.context.function_name == "outer":
            return "{ syscall.log(\"outer\"); inner(\"nested\"); return; }"
        return "{ syscall.log(msg); return; }"

    interpreter, messages = _build_interpreter(
        """plan {
            function main() : Void {
                outer();
                return;
            }

            @Deferred
            function outer() : Void;

            @Deferred
            function inner(msg: String) : Void;
        }""",
        cpl_parser,
        planner=planner,
    )

    interpreter.run()

    assert messages == ["outer", "nested"]
    assert planner_calls == ["outer", "inner"]


def test_deferred_function_without_planner_errors(cpl_parser):
    interpreter, _ = _build_interpreter(
        """plan {
            function main() : Void {
                perform();
                return;
            }

            @Deferred
            function perform() : Void;
        }""",
        cpl_parser,
    )

    with pytest.raises(CPLRuntimeError, match="Deferred function 'perform'"):
        interpreter.run()