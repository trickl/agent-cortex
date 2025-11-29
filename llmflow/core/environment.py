"""Deprecated environment shim kept for compatibility."""

raise RuntimeError(
    "llmflow.core.environment is no longer part of the CPL agent runtime."
)
from typing import Dict, Any, Callable
import json

# Assuming tool_registry.py provides these functions
from llmflow.tools.tool_registry import get_tool_function

class Environment:
    """Manages the execution of actions (tools) in the external world."""

    def __init__(self):
        """Initializes the environment."""
        # In a real application, this might include connections to databases,
        # file system handlers, API clients, etc.
        print("Environment initialized.")

    def execute_action(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Finds and executes a registered tool/action.

        Args:
            tool_name: The name of the tool to execute.
            tool_args: A dictionary of arguments for the tool.

        Returns:
            The result of the tool execution.

        Raises:
            ValueError: If the tool is not found or if arguments are incorrect (though basic validation happens here).
            Exception: Any exception raised by the tool itself during execution.
        """
        tool_function = get_tool_function(tool_name)

        if not tool_function:
            error_message = f"Environment Error: Tool '{tool_name}' not found."
            print(error_message)
            # Returning an error structure that can be JSON serialized easily
            return {"error": "ToolNotFound", "message": error_message}

        # print(f"[Environment] Attempting to execute tool: '{tool_name}' with args: {tool_args}")
        try:
            # The registered tool function is called directly with the parsed arguments.
            # The tool itself should handle its specific logic and error handling.
            result = tool_function(**tool_args)
            # print(f"[Environment] Tool '{tool_name}' executed successfully. Result: {result}")
            return result
        except TypeError as e:
            # This often happens if arguments are missing or of the wrong type, 
            # and the tool function isn't robust enough or the LLM provided bad args.
            error_message = f"Environment Error: Type error executing tool '{tool_name}\' with args '{tool_args}\'. Possible missing or incorrect arguments. Details: {e}"
            print(error_message)
            return {"error": "ToolExecutionTypeError", "message": error_message, "tool_name": tool_name, "args": tool_args}
        except Exception as e:
            # Catch-all for other exceptions during tool execution
            error_message = f"Environment Error: Exception during execution of tool '{tool_name}\' with args '{tool_args}\'. Details: {type(e).__name__} - {e}"
            print(error_message)
            return {"error": "ToolExecutionError", "message": error_message, "tool_name": tool_name, "details": str(e)}


# Example Usage (Conceptual - typically, the Agent would use this)
if __name__ == '__main__':
    # This example requires tools to be registered, so we might need to import them
    # or ensure tool_decorator's example tools ran.
    try:
        # Attempt to load example tools from tool_decorator if not already loaded
        from llmflow.tools.tool_decorator import _tool_registry, register_tool # Import register_tool for dummy
        if not _tool_registry or ("read_file" not in _tool_registry):
            print("Environment Example: Registering a dummy tool as no tools found.")
            @register_tool(tags=["environment_example"])
            def example_env_tool(location: str, unit: str = "celsius") -> Dict[str, str]:
                """A dummy tool for environment testing. Gets fake weather.
                Args:
                    location: The city name.
                    unit: Temperature unit (celsius or fahrenheit).
                Returns:
                    A dictionary with weather information.
                """
                if location.lower() == "testville":
                    return {"location": location, "temperature": f"25 {unit}", "condition": "Sunny"}
                else:
                    return {"location": location, "error": "Weather data not found"}
            
            @register_tool(tags=["environment_example", "erroring"])
            def example_erroring_tool(message: str) -> None:
                """This tool always raises an error."""
                raise ValueError(f"Intentional error from example_erroring_tool: {message}")

    except ImportError:
        print("Warning: Could not import from tool_decorator. Tool registry might be empty.")
        # Define a very simple tool if imports fail, to allow Environment to be testable
        if not get_tool_function("simple_echo_tool"): # Check if it already exists
            # This direct manipulation of _tool_registry is not ideal but helps for a standalone example
            # In a real app, ensure tools are discoverable via imports.
            _tool_registry["simple_echo_tool"] = {
                "function": lambda text: f"Echo: {text}", 
                "schema": { "type": "function", "function": {"name": "simple_echo_tool", "description": "Echoes text", "parameters": {"type":"object", "properties": {"text": {"type":"string"}}, "required":["text"]}}},
                "description": "Echoes back the provided text.",
                "tags": ["dummy"]
            }
            print("Registered simple_echo_tool for environment test.")


    env = Environment()

    print("\n--- Environment: Testing Successful Tool Execution ---")
    # Ensure the tool 'example_env_tool' or 'simple_echo_tool' is available based on imports
    tool_to_test_success = "example_env_tool" if get_tool_function("example_env_tool") else "simple_echo_tool"
    args_for_success = {"location": "Testville", "unit": "fahrenheit"} if tool_to_test_success == "example_env_tool" else {"text": "Hello World"}
    
    result1 = env.execute_action(tool_to_test_success, args_for_success)
    print(f"Result for '{tool_to_test_success}': {json.dumps(result1, indent=2)}")

    if tool_to_test_success == "example_env_tool":
        result_unknown_loc = env.execute_action("example_env_tool", {"location": "UnknownCity"})
        print(f"Result for unknown location: {json.dumps(result_unknown_loc, indent=2)}")

    print("\n--- Environment: Testing Non-Existent Tool ---")
    result2 = env.execute_action("non_existent_tool_xyz", {"arg": "value"})
    print(f"Result for non_existent_tool_xyz: {json.dumps(result2, indent=2)}")

    print("\n--- Environment: Testing Tool with TypeError (e.g. wrong/missing args not caught by tool) ---")
    # This test relies on a tool that might raise TypeError if args are wrong.
    # Let's try to call example_env_tool with missing required arg 'location'
    if get_tool_function("example_env_tool"):
        result3 = env.execute_action("example_env_tool", {"unit": "celsius"}) # Missing 'location'
        print(f"Result for example_env_tool (missing args): {json.dumps(result3, indent=2)}")
    else:
        print("(Skipped TypeError test as example_env_tool is not registered)")

    print("\n--- Environment: Testing Tool that Intentionally Raises an Error ---")
    if get_tool_function("example_erroring_tool"):
        result4 = env.execute_action("example_erroring_tool", {"message": "test failure"})
        print(f"Result for example_erroring_tool: {json.dumps(result4, indent=2)}")
    else:
        print("(Skipped Intentional Error test as example_erroring_tool is not registered)")

    print("\nEnvironment tests finished.") 