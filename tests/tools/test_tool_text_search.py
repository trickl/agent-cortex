"""Tests for the repository text search tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmflow.tools import tool_text_search


@pytest.fixture()
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated repository root for the search helper."""

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(tool_text_search, "REPO_ROOT", root)
    return root


def _seed_files(base: Path) -> None:
    (base / "notes").mkdir()
    (base / "misc" / "hidden").mkdir(parents=True)

    (base / "notes" / "clue.txt").write_text(
        "Top secret\nLook for abU7es2 inside this file.\n",
        encoding="utf-8",
    )
    (base / "notes" / "decoy.txt").write_text("Nothing to see here", encoding="utf-8")
    (base / "misc" / "binary.bin").write_bytes(b"\x00\x01\x02")
    (base / "misc" / "hidden" / ".clue.txt").write_text(
        "abU7es2 is also here", encoding="utf-8"
    )


def test_search_finds_matches(repo_root: Path) -> None:
    _seed_files(repo_root)
    result = tool_text_search.search_text_in_repository(
        search_root="notes",
        query="abU7es2",
        case_sensitive=False,
    )

    assert result["success"] is True
    assert result["results"]
    first_hit = result["results"][0]
    assert first_hit["file_path"].endswith("notes/clue.txt")
    assert first_hit["line_number"] == 2
    assert "abU7es2" in first_hit["line"]


def test_hidden_files_opt_in(repo_root: Path) -> None:
    _seed_files(repo_root)
    result = tool_text_search.search_text_in_repository(
        search_root=".",
        query="abU7es2",
        include_hidden=False,
    )
    assert len(result["results"]) == 1

    include_hidden = tool_text_search.search_text_in_repository(
        search_root=".",
        query="abU7es2",
        include_hidden=True,
        max_results=5,
    )
    assert len(include_hidden["results"]) == 2


def test_extension_filter(repo_root: Path) -> None:
    _seed_files(repo_root)
    result = tool_text_search.search_text_in_repository(
        search_root=".",
        query="abU7es2",
        allowed_extensions=[".md"],
    )
    assert result["results"] == []

    allowed = tool_text_search.search_text_in_repository(
        search_root=".",
        query="abU7es2",
        allowed_extensions=["txt"],
    )
    assert allowed["results"]


def test_validation_errors_are_reported(repo_root: Path) -> None:
    output = tool_text_search.search_text_in_repository(
        search_root=".",
        query="",
    )
    assert output["success"] is False
    assert "must not be empty" in output["error"]

    invalid_root = tool_text_search.search_text_in_repository(
        search_root="missing",
        query="anything",
    )
    assert invalid_root["success"] is False
    assert "does not exist" in invalid_root["error"]


def test_filesystem_root_maps_to_repo_root(repo_root: Path) -> None:
    _seed_files(repo_root)
    result = tool_text_search.search_text_in_repository(
        search_root="/",
        query="abU7es2",
    )

    assert result["success"] is True
    assert result["results"]
    assert result["search_root"] == "."
    first_hit = result["results"][0]
    assert first_hit["file_path"].endswith("notes/clue.txt")


def test_nonexistent_absolute_path_falls_back(repo_root: Path) -> None:
    _seed_files(repo_root)
    outside_path = repo_root.parent / "missing-directory"
    result = tool_text_search.search_text_in_repository(
        search_root=str(outside_path),
        query="abU7es2",
    )

    assert result["success"] is True
    assert result["search_root"] == "."