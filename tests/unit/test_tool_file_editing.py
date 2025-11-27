from __future__ import annotations

from pathlib import Path

from llmflow.tools import tool_file_editing


def test_overwrite_text_file_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "src" / "module.py"

    result = tool_file_editing.overwrite_text_file(
        file_path=str(target),
        content="value = 1",
        create_directories=True,
    )

    assert result["success"] is True
    assert target.read_text(encoding="utf-8").strip() == "value = 1"


def test_apply_text_rewrite_single_occurrence(tmp_path: Path) -> None:
    file_path = tmp_path / "app.py"
    file_path.write_text("status = 'todo'\n", encoding="utf-8")

    result = tool_file_editing.apply_text_rewrite(
        file_path=str(file_path),
        original_snippet="'todo'",
        new_snippet="'done'",
        occurrence=1,
    )

    assert result["success"] is True
    assert "replacements" in result
    assert file_path.read_text(encoding="utf-8").strip() == "status = 'done'"
