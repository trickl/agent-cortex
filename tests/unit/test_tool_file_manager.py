from __future__ import annotations

from pathlib import Path

from llmflow.tools import tool_file_manager


def test_list_files_in_tree(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    target = workspace / "src" / "main.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('hello world')\n", encoding="utf-8")

    result = tool_file_manager.list_files_in_tree(str(workspace), pattern="**/*.py")

    assert result["success"] is True
    assert result["count"] == 1
    entry = result["files"][0]
    assert entry["relative_path"].endswith("src/main.py")
    assert entry["is_binary"] is False


def test_read_text_file_truncation(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("abcdefghij", encoding="utf-8")

    result = tool_file_manager.read_text_file(str(file_path), max_characters=5)

    assert result["success"] is True
    assert result["content"] == "abcde"
    assert result["truncated"] is True
