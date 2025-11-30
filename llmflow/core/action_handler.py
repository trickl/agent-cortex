"""
Deprecated ActionHandler module kept for backwards compatibility imports.

The modern Java plan agent no longer executes tools directly from raw LLM
responses, so this module is intentionally disabled. Importing it raises at
import time to surface the migration requirement immediately.
"""

raise RuntimeError(
    "llmflow.core.action_handler has been removed; migrate to the Java plan pipeline."
)

import json
from uuid import uuid4
from typing import List, Dict, Callable, Any, Optional, Tuple, Mapping

# Tooling - to get the actual function to execute
from llmflow.tools.tool_registry import get_tool_function, get_tool_schema

_ERROR_HINT_FIELDS = ("error", "message", "detail", "details", "reason")
_FATAL_EXCEPTION_TYPES = (
    ModuleNotFoundError,
    ImportError,
    PermissionError,
    FileNotFoundError,
    NotADirectoryError,
)
_RETRYABLE_EXCEPTION_TYPES = (TimeoutError, ConnectionError)


def _default_metadata() -> Dict[str, Any]:
    return {"success": True, "retryable": True, "fatal": False, "error": None}


def _merge_metadata_from_mapping(metadata: Dict[str, Any], payload: Mapping[str, Any]) -> Tuple[bool, bool]:
    success_explicit = False
    retryable_explicit = False

    if "success" in payload:
        metadata["success"] = bool(payload["success"])
        success_explicit = True

    if "retryable" in payload:
        metadata["retryable"] = bool(payload["retryable"])
        retryable_explicit = True

    if "fatal" in payload:
        metadata["fatal"] = bool(payload["fatal"])

    for field in _ERROR_HINT_FIELDS:
        if field in payload and payload[field]:
            metadata["error"] = str(payload[field])
            if not success_explicit:
                metadata["success"] = False
            break

    return success_explicit, retryable_explicit


def _normalize_tool_result_payload(payload: Any) -> Tuple[str, Dict[str, Any]]:
    metadata = _default_metadata()
    success_explicit = False
    retryable_explicit = False

    if isinstance(payload, str):
        content_str = payload
        stripped = payload.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(payload)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, Mapping):
                success_flag, retry_flag = _merge_metadata_from_mapping(metadata, parsed)
                success_explicit = success_explicit or success_flag
                retryable_explicit = retryable_explicit or retry_flag
    elif isinstance(payload, (dict, list, tuple)):
        try:
            content_str = json.dumps(payload)
        except TypeError:
            content_str = str(payload)
        if isinstance(payload, Mapping):
            success_flag, retry_flag = _merge_metadata_from_mapping(metadata, payload)
            success_explicit = success_explicit or success_flag
            retryable_explicit = retryable_explicit or retry_flag
    else:
        content_str = str(payload)

    if metadata["fatal"]:
        metadata["success"] = False
        metadata["retryable"] = False
    elif not metadata["success"] and not retryable_explicit:
        metadata["retryable"] = False
    elif metadata["success"] and not retryable_explicit:
        metadata["retryable"] = True

    if metadata["error"] is None and not metadata["success"]:
        metadata["error"] = content_str[:500] if content_str else "Unknown error"

    return content_str, metadata


def _classify_exception(exc: Exception) -> Dict[str, bool]:
    fatal = isinstance(exc, _FATAL_EXCEPTION_TYPES)
    retryable = isinstance(exc, _RETRYABLE_EXCEPTION_TYPES)
    if fatal:
        retryable = False
    return {"fatal": fatal, "retryable": retryable}

class ActionHandler:
    """Parses LLM tool calls and executes them via the tool registry."""

    def __init__(self, environment_execute_action_callback: Optional[Callable[[str, Dict], Any]] = None):
        """
        Args:
            environment_execute_action_callback: A callback function to execute actions in the environment.
                                                 This is optional and currently not the primary way tools are run.
                                                 Tools are primarily run via get_tool_function.
        """

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
            if not isinstance(call, dict):
                print(f"Warning: Skipping invalid tool call object (not a dict): {call}")
                continue

            func_spec = call.get("function")
            if not isinstance(func_spec, dict):
                print(f"Warning: Skipping tool call due to missing function spec: {call}")
                continue

            func_name = func_spec.get("name")
            if not func_name:
                print(f"Warning: Tool call missing function name: {call}")
                continue

            arguments_payload = func_spec.get("arguments", {})
            if isinstance(arguments_payload, (dict, list)):
                try:
                    arguments_payload = json.dumps(arguments_payload)
                except (TypeError, ValueError) as exc:
                    print(
                        f"Warning: Could not serialize arguments for tool '{func_name}': {exc}. Skipping call."
                    )
                    continue
            elif arguments_payload is None:
                arguments_payload = "{}"
            elif not isinstance(arguments_payload, str):
                arguments_payload = json.dumps(arguments_payload)

            normalized_call = {
                "id": call.get("id") or f"auto_call_{uuid4().hex}",
                "type": call.get("type") or "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments_payload,
                },
            }

            if normalized_call["type"] != "function":
                print(f"Warning: Unsupported tool call type '{normalized_call['type']}'. Skipping call {call}.")
                continue

            action_requests.append(normalized_call)
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
                error_payload = {
                    "success": False,
                    "error": "Malformed tool call received by ActionHandler.",
                    "details": "Missing id, name, or arguments.",
                    "retryable": False,
                }
                content_str, metadata = _normalize_tool_result_payload(error_payload)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id or "unknown_id",
                    "name": tool_name or "unknown_tool",
                    "content": content_str,
                    "metadata": metadata,
                })
                continue

            try:
                tool_function = get_tool_function(tool_name)
                tool_schema_full = get_tool_schema(tool_name) # For terminal property
                is_terminal_tool = tool_schema_full.get("terminal", False) if tool_schema_full else False
                
                if not tool_function:
                    error_msg = f"Tool '{tool_name}' not found in registry."
                    print(f"Error: {error_msg}")
                    error_payload = {
                        "success": False,
                        "error": error_msg,
                        "fatal": True,
                        "retryable": False,
                    }
                    content_str, metadata = _normalize_tool_result_payload(error_payload)
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": content_str,
                        "metadata": metadata,
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
                    error_payload = {
                        "success": False,
                        "error": "Invalid JSON in arguments.",
                        "details": error_msg,
                        "retryable": False,
                    }
                    content_str, metadata = _normalize_tool_result_payload(error_payload)
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": content_str,
                        "metadata": metadata,
                    })
                    continue
                except TypeError as e_type:
                    error_msg = f"Arguments for tool '{tool_name}' did not parse to a dictionary: {arguments_str}. Error: {e_type}"
                    print(f"Error: {error_msg}")
                    error_payload = {
                        "success": False,
                        "error": "Arguments not a dictionary.",
                        "details": error_msg,
                        "retryable": False,
                    }
                    content_str, metadata = _normalize_tool_result_payload(error_payload)
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": content_str,
                        "metadata": metadata,
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

                action_result_str, metadata = _normalize_tool_result_payload(action_result)

                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": action_result_str,
                    "is_terminal": is_terminal_tool,
                    "metadata": metadata,
                })

            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                print(f"Error: {error_msg}")
                classification = _classify_exception(e)
                error_payload = {
                    "success": False,
                    "error": error_msg,
                    "exception_type": e.__class__.__name__,
                    "retryable": classification["retryable"],
                    "fatal": classification["fatal"],
                }
                content_str, metadata = _normalize_tool_result_payload(error_payload)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content_str,
                    "metadata": metadata,
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