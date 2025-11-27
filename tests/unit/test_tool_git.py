"""Unit tests for git tool helpers."""

import pytest

from llmflow.tools import tool_git


@pytest.fixture(autouse=True)
def clear_project_repo_root(monkeypatch):
    monkeypatch.delenv("PROJECT_REPO_ROOT", raising=False)


def test_resolve_clone_destination_defaults_to_repo_root(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo-root"
    monkeypatch.setenv("PROJECT_REPO_ROOT", str(repo_root))

    resolved = tool_git._resolve_clone_destination(
        "git@github.com:owner/repo.git",
        None,
    )

    expected = (repo_root / "owner" / "repo").resolve()
    assert resolved == expected
    assert resolved.parent.exists()


def test_resolve_clone_destination_respects_relative_paths(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo-root"
    monkeypatch.setenv("PROJECT_REPO_ROOT", str(repo_root))

    resolved = tool_git._resolve_clone_destination(
        "git@github.com:owner/repo.git",
        "custom/workdir",
    )

    expected = (repo_root / "custom" / "workdir").resolve()
    assert resolved == expected


def test_resolve_clone_destination_without_env(monkeypatch, tmp_path):
    target = tmp_path / "manual"

    resolved = tool_git._resolve_clone_destination(
        "git@github.com:owner/repo.git",
        str(target),
    )

    assert resolved == target.resolve()
    assert resolved.parent == target.parent.resolve()
