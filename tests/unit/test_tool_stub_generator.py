"""Tests for Java tool stub generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from llmflow.planning.tool_stub_generator import (
    ToolStubGenerationError,
    generate_tool_stub_class,
)
from llmflow.tools.tool_decorator import get_registered_tags, get_registered_tools, register_tool


@pytest.fixture(name="tool_registry_sandbox")
def tool_registry_sandbox_fixture():
    """Isolate tool registrations created within a test."""

    registry = get_registered_tools()
    tags = get_registered_tags()
    existing_names = set(registry.keys())
    yield
    for name in list(registry.keys()):
        if name not in existing_names:
            registry.pop(name, None)
    for name in list(tags.keys()):
        if name not in existing_names:
            tags.pop(name, None)


def test_generate_stub_includes_optional_parameters(tool_registry_sandbox):
    @register_tool(tags=["planning"])
    def demo_tool(path: str, retries: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Demonstration tool for stub generation."""

        return {"path": path, "retries": retries, "metadata": metadata}

    java_source = generate_tool_stub_class("DemoTools", ["demo_tool"])

    assert "public final class DemoTools" in java_source
    assert "Map<String, Object> demo_tool(String path, Integer retries, Map<String, Object> metadata)" in java_source
    assert "Optional parameters: retries, metadata." in java_source
    assert "May return null" not in java_source


def test_generate_stub_renders_collection_types(tool_registry_sandbox):
    @register_tool(tags=["planning"])
    def summarize(values: List[str]) -> List[str]:
        """Returns a normalized list of values."""

        return list(values)

    java_source = generate_tool_stub_class(
        "CollectionTools",
        ["summarize"],
        package="com.example.agents",
    )

    assert java_source.startswith("package com.example.agents;")
    assert "import java.util.List;" in java_source
    assert "public static List<String> summarize(List<String> values)" in java_source


def test_generate_stub_raises_for_unknown_tool(tool_registry_sandbox):
    with pytest.raises(ToolStubGenerationError):
        generate_tool_stub_class("MissingTools", ["does_not_exist"])
