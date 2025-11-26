"""Provides functions to access and manage registered tools."""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from llmflow.tools import (
    get_module_for_tool_name,
    get_modules_for_tags,
    load_all_tools,
    load_tool_module,
)

# Import the global registries from tool_decorator
# This creates a circular dependency if tool_decorator imports tool_registry.
# To avoid this, tool_decorator should define _tool_registry and _tool_tags,
# and this module should access them directly or via getter functions in tool_decorator.
# For simplicity here, assuming tool_decorator defines them and we import them here.
# A better structure might involve a central registry class passed around or a dedicated module for registries.

# In a real scenario, you might have a class structure or ensure that
# tool_decorator.py is imported before this module attempts to use these.
# from .tool_decorator import _tool_registry, _tool_tags # This would be typical

# For now, to make it runnable as a script and avoid complex import issues in this context,
# we'll assume _tool_registry and _tool_tags are accessible if tool_decorator was run/imported.
# This is a common pattern for simple decorator-based registries.

# If you are running tools.py directly, it will not find these. 
# They are populated when tool_decorator.py is imported and tools are defined.

# To make this module self-contained for now, let's try to get them from tool_decorator.
# This implies that tool_decorator must have been imported for this to work.
_TOOLS_FULLY_LOADED = False
LOGGER = logging.getLogger(__name__)


def _load_modules_for_tags(tags: Optional[List[str]], match_all: bool) -> None:
    global _TOOLS_FULLY_LOADED
    if _TOOLS_FULLY_LOADED:
        return
    if not tags:
        load_all_tools()
        _TOOLS_FULLY_LOADED = True
        return

    module_candidates = get_modules_for_tags(tags, match_all)
    if not module_candidates:
        LOGGER.debug(
            "No tool modules matched tags %s (match_all=%s); importing all tools as fallback.",
            tags,
            match_all,
        )
        load_all_tools()
        _TOOLS_FULLY_LOADED = True
        return

    for module_name in module_candidates:
        load_tool_module(module_name)


def _ensure_tool_loaded_by_name(tool_name: str) -> None:
    global _TOOLS_FULLY_LOADED
    if not tool_name or tool_name in _tool_registry or _TOOLS_FULLY_LOADED:
        return

    module_name = get_module_for_tool_name(tool_name)
    if module_name:
        load_tool_module(module_name)
        return

    LOGGER.debug(
        "Unknown tool '%s' not present in metadata; importing all tool modules as fallback.",
        tool_name,
    )
    load_all_tools()
    _TOOLS_FULLY_LOADED = True


try:
    from .tool_decorator import _tool_registry, _tool_tags, register_tool
except ImportError:
    # Fallback for direct execution or if structure is different
    # This means tools need to be defined in this file or tool_decorator.py needs to be imported explicitly first.
    print("Warning: tool_decorator not found using relative import. Tool registry might be empty unless tools are defined/imported elsewhere.")
    _tool_registry = {}
    _tool_tags = {}


def get_tool_function(tool_name: str) -> Optional[Callable[..., Any]]:
    """Retrieves the callable function for a given tool name."""
    _ensure_tool_loaded_by_name(tool_name)
    tool_info = _tool_registry.get(tool_name)
    return tool_info["function"] if tool_info else None

def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Retrieves the JSON schema for a given tool name."""
    _ensure_tool_loaded_by_name(tool_name)
    tool_info = _tool_registry.get(tool_name)
    return tool_info["schema"] if tool_info else None

def get_tool_description(tool_name: str) -> Optional[str]:
    """Retrieves the full docstring description for a given tool name."""
    _ensure_tool_loaded_by_name(tool_name)
    tool_info = _tool_registry.get(tool_name)
    return tool_info["description"] if tool_info else None

def get_tool_tags(tool_name: str) -> Optional[List[str]]:
    """Retrieves the tags for a given tool name."""
    # Tags are now stored directly in _tool_registry per tool
    _ensure_tool_loaded_by_name(tool_name)
    tool_info = _tool_registry.get(tool_name)
    return tool_info["tags"] if tool_info else None

def get_all_tools_schemas(tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Returns a list of JSON schemas for all registered tools, optionally filtered by tags.

    If tags are provided, only tools matching ALL provided tags are returned.
    """
    if tags:
        _load_modules_for_tags(tags, match_all=True)
    else:
        _load_modules_for_tags(tags=None, match_all=True)
    schemas = []
    for tool_name, tool_info in _tool_registry.items():
        if tags:
            # Check if all provided tags are present in the tool's tags
            tool_actual_tags = tool_info.get("tags", [])
            if all(tag in tool_actual_tags for tag in tags):
                schemas.append(tool_info["schema"])
        else:
            schemas.append(tool_info["schema"])
    return schemas

def get_tools_by_tags(tags: List[str], match_all: bool = True) -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary of tools that match the given tags.

    Args:
        tags: A list of tags to filter by.
        match_all: If True, tools must have all specified tags. 
                     If False, tools must have at least one of the specified tags.
    Returns:
        A dictionary where keys are tool names and values are their full registry entries.
    """
    _load_modules_for_tags(tags, match_all)
    if not tags: # Return all tools if no tags are specified
        return dict(_tool_registry)
        
    matched_tools = {}
    for tool_name, tool_info in _tool_registry.items():
        tool_actual_tags = tool_info.get("tags", [])
        if match_all:
            if all(tag in tool_actual_tags for tag in tags):
                matched_tools[tool_name] = tool_info
        else:
            if any(tag in tool_actual_tags for tag in tags):
                matched_tools[tool_name] = tool_info
    return matched_tools    

def list_available_tools(verbose: bool = False) -> List[str]:
    """Returns a list of names of all registered tools.
    If verbose, prints details of each tool.
    """
    _load_modules_for_tags(tags=None, match_all=True)
    if not _tool_registry:
        print("No tools are currently registered.")
        return []
        
    tool_names = list(_tool_registry.keys())
    if verbose:
        print("\nAvailable Tools Details:")
        for name in tool_names:
            info = _tool_registry[name]
            print(f"  - Tool: {name}")
            print(f"    Tags: {info.get('tags', [])}")
            print(f"    Summary: {info['schema']['function']['description']}")
    return tool_names

# Example Usage (Requires tools to be defined and registered, e.g., by importing a file that uses @register_tool)
if __name__ == '__main__':
    # To make this example runnable, we need to ensure some tools are registered.
    # This would typically happen by importing the modules where tools are defined.
    # For this standalone example, let's try to define a dummy tool here if the registry is empty.
    
    # Attempt to import tools from tool_decorator example section if they haven't been already.
    # This is a bit of a hack for making the example runnable standalone.
    try:
        # Check if example tools from tool_decorator are already in registry
        if not ('read_file' in _tool_registry and 'write_to_file' in _tool_registry):
            # This implies that tool_decorator.py might not have run its __main__ block
            # Or this script is run in a context where those definitions aren't automatically loaded.
            # For robust testing, it's better to explicitly import tool definition modules.
            print("Attempting to import example tools from tool_decorator for demonstration...")
            from .tool_decorator import read_file, write_to_file, reverse_string, add_numbers, get_current_timestamp, process_complex_data
            # The import itself should trigger registration if @register_tool is at the top level of those functions.
            # If they are inside if __name__ == '__main__' in tool_decorator.py, this won't work directly.
            # For the sake of this example, we ensure at least one tool if others are not found.
            if not _tool_registry:
                 @register_tool(tags=["dummy", "test"])
                 def my_dummy_tool(param1: str, param2: Optional[int] = 0) -> str:
                    """This is a dummy tool for testing the registry. It does nothing useful.
                    Args: param1: A string. param2: An optional int.
                    Returns: A confirmation string.
                    """
                    return f"Dummy tool executed with {param1} and {param2}"
    except ImportError as e:
        print(f"Could not import example tools from tool_decorator: {e}")
        if not _tool_registry: # Define a fallback dummy tool if import fails and registry is still empty
            @register_tool(tags=["dummy", "fallback"])
            def fallback_tool(data: str) -> str:
                """Fallback tool if others can't be loaded."""
                return f"Fallback: {data}"

    print("\n--- Tool Registry Example Usage ---")
    
    all_tool_names = list_available_tools(verbose=True)

    if not all_tool_names:
        print("\nNo tools registered to demonstrate further functionality.")
    else:
        print(f"\nAll registered tool names (summary): {all_tool_names}")

        all_schemas = get_all_tools_schemas()
        print(f"\nTotal schemas retrieved: {len(all_schemas)}")
        if all_schemas:
            print("First schema example:", json.dumps(all_schemas[0], indent=2))

        # Filter tools by a specific tag
        # Note: Ensure that tools with this tag are actually registered by tool_decorator.py's example or the dummy tool
        test_tag = "file_system" # or "dummy" if using the dummy tool
        if not get_tools_by_tags(tags=[test_tag]):
             # If file_system tools are not found, try with a tag that is likely present from dummy tool
             if _tool_registry.get('my_dummy_tool') or _tool_registry.get('fallback_tool'): 
                test_tag = "dummy" if _tool_registry.get('my_dummy_tool') else "fallback"
        
        tagged_tools_schemas = get_all_tools_schemas(tags=[test_tag])
        print(f"\nTool schemas tagged with '{test_tag}': {len(tagged_tools_schemas)}")
        for schema in tagged_tools_schemas:
            print(json.dumps(schema, indent=2))

        read_tools = get_tools_by_tags(tags=["read"])
        print(f"\nTools tagged with 'read': {list(read_tools.keys())}")
        
        # Example: Tools matching ANY of the tags ["write", "math"]
        write_or_math_tools = get_tools_by_tags(tags=["write", "math"], match_all=False)
        print(f"\nTools tagged with 'write' OR 'math': {list(write_or_math_tools.keys())}")

        # Example: Tools matching ALL of the tags ["file_system", "read"]
        fs_read_tools = get_tools_by_tags(tags=["file_system", "read"], match_all=True)
        print(f"\nTools tagged with 'file_system' AND 'read': {list(fs_read_tools.keys())}")

        if all_tool_names:
            example_tool_name = all_tool_names[0]
            print(f"\n--- Details for tool '{example_tool_name}' ---")
            print(f"  Callable Function: {get_tool_function(example_tool_name)}")
            retrieved_schema = get_tool_schema(example_tool_name)
            if retrieved_schema:
                print(f"  Schema: {json.dumps(retrieved_schema, indent=2)}")
            print(f"  Full Docstring: {get_tool_description(example_tool_name)}")
            print(f"  Registered Tags: {get_tool_tags(example_tool_name)}")

    print("\nTool registry demonstration finished.") 