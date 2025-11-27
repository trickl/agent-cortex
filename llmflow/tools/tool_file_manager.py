"""File discovery and inspection tools for agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmflow.tools.tool_decorator import register_tool

_TOOL_TAGS = ["file_system", "file_management", "file_discovery"]


@dataclass
class _FileEntry:
    absolute_path: str
    relative_path: str
    size_bytes: int
    modified_timestamp: float
    is_binary: bool


def _normalize_root(root_path: str) -> Path:
    path = Path(root_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory but received: {path}")
    return path


def _include_path(relative_parts: List[str], include_hidden: bool) -> bool:
    if include_hidden:
        return True
    for part in relative_parts:
        if part.startswith('.'):
            return False
    return True


def _is_binary(path: Path, sample_size: int = 1024) -> bool:
    try:
        with path.open("rb") as stream:
            chunk = stream.read(sample_size)
        if not chunk:
            return False
        if b"\0" in chunk:
            return True
        printable = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in {9, 10, 13})
        ratio = printable / len(chunk)
        return ratio < 0.95
    except OSError:
        return True


@register_tool(tags=_TOOL_TAGS)
def list_files_in_tree(
    root_path: str,
    pattern: Optional[str] = None,
    max_results: int = 200,
    include_hidden: bool = False,
    follow_symlinks: bool = False,
) -> Dict[str, Any]:
    """List files below a root directory with optional glob filtering."""

    try:
        root = _normalize_root(root_path)
        matched: List[_FileEntry] = []
        glob_pattern = pattern or "**/*"
        iterator = root.rglob(glob_pattern)
        for entry in iterator:
            try:
                resolved = entry if follow_symlinks else entry.resolve(strict=False)
            except OSError:
                continue
            if resolved.is_dir():
                continue
            relative_parts = list(entry.relative_to(root).parts)
            if not _include_path(relative_parts, include_hidden):
                continue
            stat = resolved.stat()
            matched.append(
                _FileEntry(
                    absolute_path=str(resolved),
                    relative_path=str(entry.relative_to(root)),
                    size_bytes=stat.st_size,
                    modified_timestamp=stat.st_mtime,
                    is_binary=_is_binary(resolved),
                )
            )
            if len(matched) >= max_results:
                break
        return {
            "success": True,
            "root": str(root),
            "count": len(matched),
            "files": [asdict(entry) for entry in matched],
            "pattern": glob_pattern,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"success": False, "error": str(exc)}


@register_tool(tags=_TOOL_TAGS)
def read_text_file(
    file_path: str,
    encoding: str = "utf-8",
    max_characters: Optional[int] = None,
) -> Dict[str, Any]:
    """Read a UTF-8 (or user-provided encoding) text file for analysis."""

    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Expected file path, received directory: {path}")
        text = path.read_text(encoding=encoding)
        truncated = False
        if max_characters is not None and len(text) > max_characters:
            text = text[:max_characters]
            truncated = True
        return {
            "success": True,
            "path": str(path),
            "encoding": encoding,
            "truncated": truncated,
            "content": text,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}