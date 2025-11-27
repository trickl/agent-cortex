"""
Decorator for easily registering tools and generating their JSON schema.
"""
import inspect
import json
import logging
from typing import Callable, Dict, Any, List, get_type_hints, Union, Optional

# These are global registries. In a more complex application, you might wrap them in a class.
_tool_registry: Dict[str, Dict[str, Any]] = {}
_tool_tags: Dict[str, List[str]] = {}  # Stores tool_name -> list_of_tags
LOGGER = logging.getLogger(__name__)


def register_tool(tags: Optional[List[str]] = None, terminal: bool = False):
    """A decorator to register a function as an agent tool and generate its schema.

    The decorated function must have type hints for all its arguments and its return type.
    The docstring will be used as the description of the tool.
    The first line of the docstring is the summary, the rest is the detailed description.

    Args:
        tags (Optional[List[str]]): A list of tags to categorize the tool.
        terminal (bool): If True, indicates that calling this tool should terminate the agent's current run.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        global _tool_registry
        global _tool_tags

        tool_name = func.__name__
        if tool_name in _tool_registry:
            LOGGER.warning("Tool '%s' is already registered. Overwriting.", tool_name)

        docstring = inspect.getdoc(func)
        if not docstring:
            raise ValueError(f"Tool function '{tool_name}' must have a docstring for its description.")

        docstring_lines = docstring.strip().split('\n', 1)
        summary = docstring_lines[0]

        type_hints = get_type_hints(func)  # Get evaluated type hints (handles future annotations)
        parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        sig = inspect.signature(func)

        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'cls':  # Skip self/cls for methods
                continue

            arg_type = type_hints.get(param_name, param.annotation)
            if arg_type is inspect.Parameter.empty:
                raise ValueError(
                    f"Argument '{param_name}' in tool '{tool_name}' is missing a type hint."
                )

            param_schema = _map_type_to_json_schema(arg_type, tool_name, param_name)

            parameters_schema["properties"][param_name] = param_schema
            if param.default is inspect.Parameter.empty:
                parameters_schema["required"].append(param_name)

        if not parameters_schema["required"] and parameters_schema["properties"]:
            del parameters_schema["required"]
        elif not parameters_schema["properties"]:
            parameters_schema = {"type": "object", "properties": {}}

        tool_schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": summary,
                "parameters": parameters_schema,
            }
        }

        current_tags = list(set(tags)) if tags else []
        _tool_registry[tool_name] = {
            "schema": tool_schema,
            "function": func,
            "description": docstring,
            "tags": current_tags,
            "terminal": terminal
        }

        # Update the separate _tool_tags mapping
        _tool_tags[tool_name] = current_tags

        return func

    return decorator


def _map_type_to_json_schema(py_type: Any, tool_name: str, arg_name: str) -> Dict[str, Any]:
    """Maps Python types to JSON schema type definitions."""
    origin = getattr(py_type, '__origin__', None)
    args = getattr(py_type, '__args__', [])

    if py_type is str:
        return {"type": "string"}
    elif py_type is int:
        return {"type": "integer"}
    elif py_type is float:
        return {"type": "number"}
    elif py_type is bool:
        return {"type": "boolean"}
    elif py_type is list or origin is list:
        if args and len(args) == 1:
            item_schema = _map_type_to_json_schema(args[0], tool_name, f"{arg_name} items")
            return {"type": "array", "items": item_schema}
        return {"type": "array", "items": {"description": "Items of the list. Type not specified."}}
    elif py_type is dict or origin is dict:
        if args and len(args) == 2:  # Handles Dict[KeyType, ValueType]
            value_type_schema = _map_type_to_json_schema(args[1], tool_name, f"{arg_name} values")
            return {"type": "object", "additionalProperties": value_type_schema}
        return {"type": "object", "additionalProperties": True}
    elif origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _map_type_to_json_schema(non_none_args[0], tool_name, arg_name)
        elif len(non_none_args) > 1:
            first_type_schema = _map_type_to_json_schema(non_none_args[0], tool_name, arg_name)
            type_names = ", ".join(str(t) for t in non_none_args)
            description_addition = f" (Value can be one of: {type_names}. Schema shown for {non_none_args[0]}.)"
            first_type_schema["description"] = first_type_schema.get("description", "") + description_addition
            LOGGER.debug(
                "Union type for %s in %s mapped to first type (%s) with extended description.",
                arg_name,
                tool_name,
                non_none_args[0],
            )
            return first_type_schema
        else:
            raise ValueError(
                f"Argument '{arg_name}' in tool '{tool_name}' has a Union type with only NoneType, which is not valid."
            )
    elif py_type is Any:
        return {"description": "Any value is permissible."}
    else:
        LOGGER.debug(
            "Unsupported type '%s' for argument '%s' in tool '%s'. Defaulting to string.",
            py_type,
            arg_name,
            tool_name,
        )
        return {
            "type": "string",
            "description": f"Represents a '{str(py_type)}' type. Provide as a string if unsure or if it's a complex object.",
        }


def get_registered_tools() -> Dict[str, Dict[str, Any]]:
    return _tool_registry


def get_registered_tags() -> Dict[str, List[str]]:
    return _tool_tags


if __name__ == '__main__':
    # This block is now only for demonstration/testing if this file is run directly

    @register_tool(tags=["file_system", "read", "example_tool"])
    def read_file_example(path: str, encoding: Optional[str] = "utf-8") -> str:
        """Reads a file and returns its content as a string (EXAMPLE IMPLEMENTATION).

        This tool simulates reading a file from the filesystem.
        Args:
            path: The absolute or relative path to the file.
            encoding: The file encoding to use (e.g., 'utf-8', 'ascii'). Defaults to 'utf-8'.
        Returns:
            The content of the file.
        """
        print(f"[EXAMPLE TOOL] Simulating: Reading file '{path}' with encoding '{encoding}'.")
        return f"Content of {path} (simulated by example_tool)"

    @register_tool(tags=["file_system", "write", "example_tool"])
    def write_to_file_example(file_path: str, text_content: str, mode: Optional[str] = "w") -> Dict[str, Any]:
        """Writes text content to a specified file (EXAMPLE IMPLEMENTATION).

        Simulates writing to a file. Supports different write modes.
        Args:
            file_path: The path of the file to write to.
            text_content: The string content to write.
            mode: File opening mode ('w' for write/overwrite, 'a' for append). Defaults to 'w'.
        Returns:
            A dictionary with a status message and the path of the file.
        """
        print(f"[EXAMPLE TOOL] Simulating: Writing to file '{file_path}' in mode '{mode}'. Content: '{text_content[:30]}...'")
        return {"status": "success", "path": file_path, "chars_written": len(text_content)}

    @register_tool(tags=["utility", "string_ops", "example_tool"])
    def reverse_string_example(text: str) -> str:
        """Reverses a given string (EXAMPLE IMPLEMENTATION).
        Args:
            text: The string to be reversed.
        Returns:
            The reversed string.
        """
        print(f"[EXAMPLE TOOL] Simulating: Reversing string '{text}'.")
        return text[::-1]

    @register_tool(tags=["math", "calculation", "example_tool"])
    def add_numbers_example(a: int, b: int = 10) -> int:
        """Adds two integers. The second integer defaults to 10 if not provided (EXAMPLE IMPLEMENTATION).
        Args:
            a: The first integer.
            b: The second integer, defaults to 10.
        Returns:
            The sum of a and b.
        """
        print(f"[EXAMPLE TOOL] Simulating: Adding {a} + {b}")
        return a + b

    @register_tool(tags=["system_info", "example_tool"])
    def get_current_timestamp_example() -> str:
        """Returns the current system timestamp as an ISO format string (simulated) (EXAMPLE IMPLEMENTATION).
        Returns:
            A string representing the current timestamp.
        """
        import datetime
        ts = datetime.datetime.now().isoformat()
        print(f"[EXAMPLE TOOL] Simulating: Getting current timestamp. Result: {ts}")
        return ts

    @register_tool(tags=["test_complex_types", "example_tool"])
    def process_complex_data_example(config: Dict[str, Union[int, str]], items: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Processes complex data structures (EXAMPLE IMPLEMENTATION).
        Args:
            config: A configuration dictionary with string keys and int or string values.
            items: An optional list of dictionaries, where each dictionary can have any structure.
        Returns:
            True if processing was (simulated) successful.
        """
        print(f"[EXAMPLE TOOL] Simulating: Processing complex data. Config: {config}, Items: {items}")
        return True

    print("--- Tool Registration Examples (from tool_decorator.py direct run) ---")
    print("\nRegistered Tools Details:")
    for tool_name, tool_data in _tool_registry.items():
        print(f"\nTool: {tool_name}")
        print(f"  Tags: {tool_data['tags']}")
        print(f"  Description (Summary): {tool_data['schema']['function']['description']}")
        print(f"  Schema: {json.dumps(tool_data['schema'], indent=2)}")

    print("\n--- Tags Information (_tool_tags mapping) ---")
    print(json.dumps(_tool_tags, indent=2))

    llm_tools_list = [info["schema"] for info in _tool_registry.values()]
    print("\n--- Schemas List for LLM API ---")
    print(json.dumps(llm_tools_list, indent=2))

    print(f"\nSchema for process_complex_data: {json.dumps(_tool_registry['process_complex_data_example']['schema'], indent=2)}")
