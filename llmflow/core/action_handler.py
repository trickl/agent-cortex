"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Action Handler Module - Core component managing the execution of agent actions.
This module implements the Actions component of the GAME methodology:
- Parses LLM responses to identify tool calls
- Validates and processes tool arguments
- Executes tools through the tool registry
- Handles tool execution errors and results
- Supports both direct tool execution and environment-mediated actions
- Manages terminal tools that can end agent execution
- Provides comprehensive error handling and reporting

The Action Handler ensures reliable and safe execution of agent-requested actions
while maintaining proper communication format between the LLM and tools.
"""

import json
from typing import List, Dict, Callable, Any, Optional

# Tooling - to get the actual function to execute
from llmflow.tools.tool_registry import get_tool_function, get_tool_schema

class ActionHandler:
    """Parses LLM tool calls and executes them via the tool registry."""

    def __init__(self, environment_execute_action_callback: Optional[Callable[[str, Dict], Any]] = None):
        """
        Args:
            environment_execute_action_callback: A callback function to execute actions in the environment.
                                                 This is optional and currently not the primary way tools are run.
                                                 Tools are primarily run via get_tool_function.
        """
        self.environment_execute_action_callback = environment_execute_action_callback

    def parse_llm_response_for_tool_calls(self, assistant_message_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts tool call requests from an assistant's message object.

        Args:
            assistant_message_obj: The assistant's message dictionary, which might contain 'tool_calls'.
                                   Example: {"role": "assistant", "content": null, "tool_calls": [...]}

        Returns:
            A list of tool call dictionaries, each representing a requested action.
            Example: [{"id": "call_123", "type": "function", "function": {"name": "tool_name", "arguments": "{\"arg\": \"val\"}"}}]
                     Returns an empty list if no valid tool calls are found.
        """
        tool_calls = assistant_message_obj.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            return []

        action_requests = []
        for call in tool_calls:
            if not isinstance(call, dict) or call.get("type") != "function":
                print(f"Warning: Skipping invalid tool call object (not a dict or type is not function): {call}")
                continue
            
            func_spec = call.get("function")
            if not isinstance(func_spec, dict) or "name" not in func_spec or "arguments" not in func_spec:
                print(f"Warning: Skipping invalid function specification in tool call: {func_spec}")
                continue

            action_requests.append(call) # Store the original tool_call structure
        return action_requests

    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes a list of tool calls and returns their results.

        Args:
            tool_calls: A list of tool call dictionaries (from parse_llm_response_for_tool_calls).

        Returns:
            A list of tool result dictionaries, ready to be added to agent memory.
            Example: [{"role": "tool", "tool_call_id": "call_123", "name": "tool_name", "content": "result_string_or_json"}]
        """
        results = []
        for tool_call_item in tool_calls:
            tool_call_id = tool_call_item.get("id")
            function_spec = tool_call_item.get("function", {})
            tool_name = function_spec.get("name")
            arguments_str = function_spec.get("arguments")

            if not tool_call_id or not tool_name or arguments_str is None: # arguments_str can be empty string for no-arg functions
                print(f"Warning: Skipping tool call due to missing id, name, or arguments: {tool_call_item}")
                results.append({
                    "role": "tool", 
                    "tool_call_id": tool_call_id or "unknown_id", 
                    "name": tool_name or "unknown_tool",
                    "content": json.dumps({"error": "Malformed tool call received by ActionHandler.", "details": "Missing id, name, or arguments."})
                })
                continue

            try:
                tool_function = get_tool_function(tool_name)
                tool_schema_full = get_tool_schema(tool_name) # For terminal property
                is_terminal_tool = tool_schema_full.get("terminal", False) if tool_schema_full else False
                
                if not tool_function:
                    error_msg = f"Tool '{tool_name}' not found in registry."
                    print(f"Error: {error_msg}")
                    results.append({
                        "role": "tool", 
                        "tool_call_id": tool_call_id, 
                        "name": tool_name, 
                        "content": json.dumps({"error": error_msg})
                    })
                    continue

                try:
                    # Ensure arguments_str is valid JSON, even if it's an empty object string "{}"
                    if not arguments_str.strip(): # Handles empty or whitespace-only strings for no-arg functions
                        args_dict = {}
                    else:
                        args_dict = json.loads(arguments_str)
                    
                    if not isinstance(args_dict, dict):
                        raise TypeError("Parsed arguments are not a dictionary.")
                        
                except json.JSONDecodeError as e_json:
                    error_msg = f"Invalid JSON arguments for tool '{tool_name}': {arguments_str}. Error: {e_json}"
                    print(f"Error: {error_msg}")
                    results.append({
                        "role": "tool", 
                        "tool_call_id": tool_call_id, 
                        "name": tool_name, 
                        "content": json.dumps({"error": "Invalid JSON in arguments.", "details": error_msg})
                    })
                    continue
                except TypeError as e_type:
                    error_msg = f"Arguments for tool '{tool_name}' did not parse to a dictionary: {arguments_str}. Error: {e_type}"
                    print(f"Error: {error_msg}")
                    results.append({
                        "role": "tool", 
                        "tool_call_id": tool_call_id, 
                        "name": tool_name, 
                        "content": json.dumps({"error": "Arguments not a dictionary.", "details": error_msg})
                    })
                    continue

                if self.environment_execute_action_callback and tool_name.startswith("env."):
                    # This is an older way of thinking about env actions, might be deprecated or unused.
                    # Modern tool usage often has specific tools for file I/O etc.
                    print(f"[ActionHandler] Executing environment action: {tool_name} with args {args_dict} via callback.")
                    action_result = self.environment_execute_action_callback(tool_name, args_dict)
                else:
                    print(f"[ActionHandler] Executing tool: {tool_name} with args {args_dict}")
                    action_result = tool_function(**args_dict)

                # Ensure result is JSON serializable (string or dict/list that can be dumped)
                # Tools should ideally return str or Dict/List.
                if not isinstance(action_result, (str, dict, list, tuple, int, float, bool, type(None))):
                    print(f"Warning: Tool '{tool_name}' returned a non-standard type: {type(action_result)}. Attempting to convert to string.")
                    try:
                        action_result_str = str(action_result)
                    except Exception as e_str_conv:
                        print(f"Error converting complex tool result for '{tool_name}' to string: {e_str_conv}")
                        action_result_str = json.dumps({"error": f"Result from tool '{tool_name}' is not directly JSON serializable and str() conversion failed.", "type": str(type(action_result))})
                elif isinstance(action_result, (dict, list, tuple)):
                     # Try to dump to JSON string if it's a collection, to ensure it's one item for 'content'
                    try:
                        action_result_str = json.dumps(action_result)
                    except TypeError as e_json_dump:
                        print(f"Error serializing result of tool '{tool_name}' to JSON: {e_json_dump}. Falling back to str().")
                        action_result_str = str(action_result)
                else: # Simple types like str, int, float, bool, None
                    action_result_str = str(action_result) # str() handles None to "None"
                
                results.append({
                    "role": "tool", 
                    "tool_call_id": tool_call_id, 
                    "name": tool_name, 
                    "content": action_result_str, # Store the potentially JSON stringified result
                    "is_terminal": is_terminal_tool # Pass terminal status along
                })

            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                print(f"Error: {error_msg}")
                results.append({
                    "role": "tool", 
                    "tool_call_id": tool_call_id, 
                    "name": tool_name, 
                    "content": json.dumps({"error": error_msg, "exception_type": e.__class__.__name__})
                })
        return results

# Example usage (illustrative)
if __name__ == '__main__':
    # This example assumes you have a tool_registry with some tools for testing.
    # And that the tool functions can be called directly.

    # 1. Register some dummy tools for this example to work directly
    # (In a real scenario, tools are registered by importing their modules)
    from llmflow.tools.tool_decorator import register_tool
    from llmflow.tools.tool_registry import _tool_registry # Access for direct check

    if "get_weather_example" not in _tool_registry:
        @register_tool(tags=["weather", "example"])
        def get_weather_example(location: str, unit: str = "celsius") -> Dict[str, str]:
            """Gets the current weather for a location (example)."""
            print(f"[Example Tool] get_weather_example called for {location} in {unit}")
            if location.lower() == "london":
                return {"location": location, "temperature": f"15 {unit}", "condition": "Cloudy"}
            return {"location": location, "temperature": "unknown", "condition": "unknown"}

    if "terminate_example" not in _tool_registry:
        @register_tool(tags=["system", "example"], terminal=True) # Mark as terminal
        def terminate_example(message: str) -> str:
            """Terminates the process with a message (example)."""
            print(f"[Example Tool] terminate_example called with: {message}")
            return f"Termination signal received. Final message: {message}"

    print("Registered tools for example:", list(_tool_registry.keys()))

    # 2. Create an ActionHandler
    handler = ActionHandler()

    # 3. Simulate an LLM response with tool calls
    simulated_llm_response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather_example",
                    "arguments": '''{"location": "london", "unit": "metric"}''' # metric is not celsius, tool uses default
                }
            },
            {
                "id": "call_def456",
                "type": "function",
                "function": {
                    "name": "non_existent_tool",
                    "arguments": '''{"param": "value"}'''
                }
            },
            {
                "id": "call_ghi789",
                "type": "function",
                "function": {
                    "name": "get_weather_example",
                    "arguments": '''{"location": "paris"}''' # Uses default unit celsius
                }
            },
            {
                "id": "call_jkl012",
                "type": "function",
                "function": {
                    "name": "terminate_example",
                    "arguments": '''{"message": "All tasks completed successfully by example."}'''
                }
            }
        ]
    }

    # 4. Parse for tool calls
    tool_call_requests = handler.parse_llm_response_for_tool_calls(simulated_llm_response)
    print("\nParsed Tool Call Requests:")
    print(json.dumps(tool_call_requests, indent=2))

    # 5. Execute tool calls
    tool_results = handler.execute_tool_calls(tool_call_requests)
    print("\nTool Execution Results:")
    print(json.dumps(tool_results, indent=2))

    print("\n--- Example with malformed arguments ---")
    malformed_llm_response = {
        "role": "assistant", "content": None,
        "tool_calls": [{
            "id": "call_mal1", "type": "function",
            "function": {"name": "get_weather_example", "arguments": '''{"location": "berlin", "unit": celsius}''' } # Invalid JSON for args
        }]
    }
    malformed_requests = handler.parse_llm_response_for_tool_calls(malformed_llm_response)
    malformed_results = handler.execute_tool_calls(malformed_requests)
    print(json.dumps(malformed_results, indent=2))

    print("\n--- Example with no-arg tool (if one was registered) ---")
    # For a hypothetical no_arg_tool registered elsewhere:
    # no_arg_llm_response = {
    #     "role": "assistant", "content": None,
    #     "tool_calls": [{"id": "call_noarg1", "type": "function", "function": {"name": "no_arg_tool", "arguments": "{}"}}]
    # }
    # no_arg_requests = handler.parse_llm_response_for_tool_calls(no_arg_llm_response)
    # no_arg_results = handler.execute_tool_calls(no_arg_requests)
    # print(json.dumps(no_arg_results, indent=2)) 