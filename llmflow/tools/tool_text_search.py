"""Utilities for searching text within the repository."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from llmflow.tools.tool_decorator import register_tool

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_FILE_BYTES = 1_000_000  # 1 MB safety cap
EXCLUDED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "venv",
    ".venv",
    "env",
}


@dataclass
class SearchStats:
    """Book-keeping for the search run."""

    scanned_files: int = 0
    skipped_files: int = 0


def _resolve_search_root(search_root: str) -> Path:
    if not search_root:
        raise ValueError("search_root cannot be empty.")

    candidate = Path(search_root).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        # Absolute paths outside the repository are clamped back to the
        # repository root so LLM-provided paths like "/" or "/app" remain
        # safe while still producing useful results.
        candidate = REPO_ROOT

    if not candidate.exists():
        raise FileNotFoundError(f"The search root '{candidate}' does not exist.")
    if not candidate.is_dir():
        raise NotADirectoryError(
            f"The search root '{candidate}' must be a directory."
        )

    return candidate


def _normalize_extensions(extensions: Optional[List[str]]) -> Optional[List[str]]:
    if not extensions:
        return None
    normalized = []
    for item in extensions:
        if not item:
            continue
        ext = item if item.startswith('.') else f".{item}"
        normalized.append(ext.lower())
    return normalized or None


def _should_skip_dir(dirname: str, include_hidden: bool) -> bool:
    if dirname in EXCLUDED_DIRECTORIES:
        return True
    if not include_hidden and dirname.startswith('.'):
        return True
    return False


def _should_skip_file(path: Path, include_hidden: bool, allowed_exts: Optional[List[str]]) -> bool:
    if not include_hidden and path.name.startswith('.'):
        return True
    if allowed_exts and path.suffix.lower() not in allowed_exts:
        return True
    try:
        size = path.stat().st_size
    except OSError:
        return True
    return size > DEFAULT_MAX_FILE_BYTES


def _search_file(
    path: Path,
    query: str,
    case_sensitive: bool,
    stats: SearchStats,
    max_hits: int,
    results: List[Dict[str, object]],
) -> None:
    stats.scanned_files += 1
    needle = query if case_sensitive else query.lower()

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                haystack = raw_line if case_sensitive else raw_line.lower()
                if needle in haystack:
                    relative_path = path.relative_to(REPO_ROOT)
                    results.append(
                        {
                            "file_path": str(relative_path),
                            "line_number": line_number,
                            "line": raw_line.rstrip('\n'),
                        }
                    )
                    if len(results) >= max_hits:
                        return
    except (UnicodeDecodeError, OSError):
        stats.skipped_files += 1


@register_tool(tags=["file_system", "search", "analysis"])
def search_text_in_repository(
    search_root: str,
    query: str,
    case_sensitive: bool = False,
    max_results: int = 25,
    include_hidden: bool = False,
    allowed_extensions: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Search for a text token within files under a repository-relative directory.

    Args:
        search_root: Directory to scan, relative to the repository root. Absolute
            paths are allowed only if they stay within the repository boundary.
            Passing the filesystem root (e.g. "/" or "C:\\") is treated as a
            shorthand for the repository root.
        query: The exact string to look for. Matching uses substring semantics.
        case_sensitive: Whether to perform a case-sensitive comparison. Defaults
            to False (case-insensitive).
        max_results: Maximum number of matches to return.
        include_hidden: If True, dot-directories and files are scanned as well.
        allowed_extensions: Optional list of file extensions to include. Entries
            may be specified with or without the leading dot (e.g. "txt" or
            ".py").

    Returns:
        A dictionary describing the outcome, including a list of matches with the
        file path (relative to repo root), line number, and line content.
    """

    if not query:
        return {
            "success": False,
            "error": "query must not be empty.",
            "results": [],
        }

    if max_results <= 0:
        return {
            "success": False,
            "error": "max_results must be greater than zero.",
            "results": [],
        }

    try:
        root_path = _resolve_search_root(search_root)
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        return {
            "success": False,
            "error": str(exc),
            "results": [],
        }

    normalized_exts = _normalize_extensions(allowed_extensions)
    results: List[Dict[str, object]] = []
    stats = SearchStats()

    for current_root, dirs, files in os.walk(root_path):
        dirs[:] = [
            d
            for d in dirs
            if not _should_skip_dir(d, include_hidden)
        ]

        for filename in files:
            path = Path(current_root) / filename
            if _should_skip_file(path, include_hidden, normalized_exts):
                stats.skipped_files += 1
                continue

            _search_file(
                path=path,
                query=query,
                case_sensitive=case_sensitive,
                stats=stats,
                max_hits=max_results,
                results=results,
            )

            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    relative_root = root_path.relative_to(REPO_ROOT)
    return {
        "success": bool(results),
        "query": query,
        "case_sensitive": case_sensitive,
        "search_root": str(relative_root),
        "results": results,
        "max_results": max_results,
        "scanned_files": stats.scanned_files,
        "skipped_files": stats.skipped_files,
    }
