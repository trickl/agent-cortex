from pathlib import Path

import pytest

from dsl.cpl_inerpreter import load_cpl_parser, parse_cpl
from dsl.cpl_validator import PlanValidator, ValidationError


GRAMMAR_PATH = Path(__file__).resolve().parents[2] / "dsl" / "cpl.lark"


@pytest.fixture(scope="session")
def cpl_parser():
    return load_cpl_parser(str(GRAMMAR_PATH))


def _validate(source: str, parser, syscalls: set[str]):
    plan = parse_cpl(source, parser)
    validator = PlanValidator(available_syscalls=syscalls)
    validator.validate(plan)


def test_validator_requires_main(cpl_parser):
    with pytest.raises(ValidationError) as exc_info:
        _validate(
            """plan {
                function helper() : Void {
                    return;
                }
            }
            """,
            cpl_parser,
            set(),
        )

    assert "main" in str(exc_info.value)


def test_validator_detects_missing_syscall(cpl_parser):
    with pytest.raises(ValidationError) as exc_info:
        _validate(
            """plan {
                function main() : Void {
                    syscall.log("hi");
                    return;
                }
            }
            """,
            cpl_parser,
            set(),
        )

    assert "Syscall 'log' not registered" in str(exc_info.value)


def test_validator_flags_undefined_variable(cpl_parser):
    with pytest.raises(ValidationError) as exc_info:
        _validate(
            """plan {
                function main() : Void {
                    syscall.log(msg);
                    return;
                }
            }
            """,
            cpl_parser,
            {"log"},
        )

    assert "'msg' not defined" in str(exc_info.value)


def test_validator_enforces_statement_limit(cpl_parser):
    body = "\n".join(
        ["                syscall.log(\"x\");" for _ in range(8)]
    )
    program = f"""plan {{
        function main() : Void {{
{body}
            return;
        }}
    }}
    """

    with pytest.raises(ValidationError) as exc_info:
        _validate(program, cpl_parser, {"log"})

    assert "exceeds" in str(exc_info.value)


def test_validator_requires_non_void_return_value(cpl_parser):
    with pytest.raises(ValidationError) as exc_info:
        _validate(
            """plan {
                function main() : Void {
                    return;
                }

                function compute() : Int {
                    let x: Int = 5;
                    return;
                }
            }
            """,
            cpl_parser,
            set(),
        )

    assert "must return a value" in str(exc_info.value)
