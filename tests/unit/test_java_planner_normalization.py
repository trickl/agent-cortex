"""Tests for Java source normalization helpers."""
from __future__ import annotations

import pytest

from llmflow.planning.java_planner import _normalize_java_source


def test_normalize_handles_markdown_headings_and_unclosed_fence() -> None:
    raw = (
        "# Java Planning Specification\n"
        "# Java Planning Specification\n\n"
        "```java\n"
        "public class Plan {\n"
        "    public void main() {\n"
        "        System.out.println(\"hi\");\n"
        "    }\n"
        "}\n"
        "``\n"
    )

    normalized = _normalize_java_source(raw)

    assert normalized.startswith("public class Plan")
    assert normalized.endswith("}")


def test_normalize_prefers_java_code_block_when_multiple_exist() -> None:
    raw = (
        "```python\nprint('noop')\n```\n"
        "```java\npublic class Planner {\n    public void main() {\n        System.out.println(\"hi\");\n    }\n}\n```\n"
    )

    normalized = _normalize_java_source(raw)

    assert "class Planner" in normalized
    assert "print('noop')" not in normalized


def test_normalize_trims_markdown_prefix_without_code_fence() -> None:
    raw = (
        "## Summary\n\n"
        "Steps to follow:\n"
        "1. Do the thing.\n"
        "2. Ship it.\n\n"
        "public class SoloPlan {\n"
        "    public void main() {\n"
        "        System.out.println(\"hi\");\n"
        "    }\n"
        "}\n"
    )

    normalized = _normalize_java_source(raw)

    assert normalized.startswith("public class SoloPlan")
    assert "Steps to follow" not in normalized


def test_normalize_decodes_escaped_newlines() -> None:
    raw = (
        "public class Planner {\\n"
        "    public static void main(String[] args) throws Exception {\\n"
        "        System.out.println(\"hi\");\\n"
        "    }\\n"
        "}"
    )

    normalized = _normalize_java_source(raw)

    assert normalized.startswith("public class Planner")
    assert normalized.splitlines()[0] == "public class Planner {"
    assert len(normalized.splitlines()) > 1


def test_normalize_best_effort_decodes_when_trailing_backslash_present() -> None:
    raw = (
        "public class Planner {\\n"
        "    public static void main(String[] args) {\\n"
        "        String repoPath = \\\"/tmp/repo\\\";\\n"
        "    }\\n"
        "}\\"
    )

    normalized = _normalize_java_source(raw)

    assert normalized.startswith("public class Planner")
    assert "\\n" not in normalized
    assert '\\"' not in normalized
    assert normalized.splitlines()[0] == "public class Planner {"


def test_normalize_rejects_placeholder_ellipsis() -> None:
    raw = "public class Planner { ... }"

    with pytest.raises(ValueError):
        _normalize_java_source(raw)


def test_normalize_rejects_insufficient_meaningful_lines() -> None:
    raw = "public class Planner {\n}\n"

    with pytest.raises(ValueError):
        _normalize_java_source(raw)
