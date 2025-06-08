"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Memory Module - Core component managing conversation history and context.
This module implements the Memory component of the GAME methodology:
- Message Class: Represents individual conversation entries with validation
  * Supports system, user, assistant, and tool messages
  * Handles tool calls and their results
  * Maintains OpenAI API compatibility
  * Provides robust validation and serialization

- Memory Class: Manages the complete conversation history
  * Maintains chronological message order
  * Supports various message types and formats
  * Handles tool/function call patterns
  * Provides flexible history access and management
  * Supports both modern and legacy function calling patterns
  * Ensures proper message validation and formatting

The Memory module ensures reliable conversation state management and
provides the context necessary for the agent to maintain coherent
interactions while supporting both modern and legacy API patterns.
"""
from typing import List, Dict, Any, Literal, Optional
import json

# Define a type for the message role according to OpenAI's API
# 'function' is kept for some legacy compatibility but 'tool' is preferred for results.
MessageRole = Literal["system", "user", "assistant", "tool", "function"]

class Message:
    """Represents a single message in the conversation history."""
    def __init__(self, role: MessageRole, content: Optional[str] = None, name: Optional[str] = None, tool_calls: Optional[List[Dict]] = None, tool_call_id: Optional[str] = None):
        if role == "assistant" and content is None and tool_calls is None:
            raise ValueError("Assistant message must have content or tool_calls.")
        if role != "assistant" and tool_calls is not None:
            raise ValueError(f"Tool calls are only allowed for 'assistant' role, not '{role}'.")
        if role == "tool" and tool_call_id is None:
            raise ValueError("Message with role 'tool' must have a tool_call_id.")
        if role == "tool" and content is None:
            # Content for a tool message is the result of the tool call
            raise ValueError("Message with role 'tool' must have content (the result of the tool call).")
        if role == "function" and name is None:
            raise ValueError("Message with role 'function' must have a name (the function name).")

        self.role = role
        self.content = content
        self.name = name
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id

    def to_dict(self) -> Dict[str, Any]:
        """Converts the message to a dictionary, suitable for LLM APIs like OpenAI."""
        msg_dict = {"role": self.role}
        if self.content is not None:
            msg_dict["content"] = self.content
        
        if self.role == "assistant" and self.tool_calls:
            msg_dict["tool_calls"] = self.tool_calls
        
        if self.role == "tool":
            if self.tool_call_id:
                msg_dict["tool_call_id"] = self.tool_call_id
            if self.name:
                msg_dict["name"] = self.name
            # OpenAI expects the name of the function that was called to be part of the 'tool' message content or a separate field
            # For simplicity, we assume content for 'tool' role includes necessary result details.
            # The 'name' field in a 'tool' message is not standard in recent OpenAI versions, but might be used by 'function' role.

        if self.role == "function" and self.name:
            msg_dict["name"] = self.name # For legacy function role

        return msg_dict

    def __repr__(self):
        parts = [f"role='{self.role}'"]
        if self.content is not None:
            parts.append(f"content='{self.content[:50]}{'...' if len(self.content) > 50 else ''}'")
        if self.name is not None:
            parts.append(f"name='{self.name}'")
        if self.tool_calls is not None:
            parts.append(f"tool_calls={self.tool_calls}")
        if self.tool_call_id is not None:
            parts.append(f"tool_call_id='{self.tool_call_id}'")
        return f"Message({', '.join(parts)})"

class Memory:
    """Stores and manages the conversation history."""
    def __init__(self, system_prompt: Optional[str] = None, initial_messages: Optional[List[Message]] = None):
        self.messages: List[Message] = []
        if system_prompt:
            self.add_system_message(system_prompt)
        if initial_messages:
            for msg in initial_messages:
                # Ensure we are adding valid Message objects if they come from external source
                if isinstance(msg, Message):
                    self.messages.append(msg)
                elif isinstance(msg, dict):
                    # Attempt to reconstruct message if dict is provided
                    try:
                        m_obj = Message(**msg)
                        self.messages.append(m_obj)
                    except Exception as e:
                        print(f"Warning: Could not reconstruct message from dict: {msg}. Error: {e}")
                else:
                    print(f"Warning: Skipping invalid initial message: {msg}")

    def add_message(self, message: Message):
        """Adds a pre-constructed Message object to the history."""
        self.messages.append(message)

    def add_system_message(self, content: str):
        """Adds a system message."""
        self.add_message(Message(role="system", content=content))

    def add_user_message(self, content: str):
        """Adds a user message."""
        self.add_message(Message(role="user", content=content))

    def add_assistant_message(self, content: Optional[str] = None, tool_calls: Optional[List[Dict]] = None):
        """Adds an assistant message. Can include text content, tool calls, or both."""
        self.add_message(Message(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool_result_message(self, tool_call_id: str, result: Any, function_name: Optional[str]=None):
        """Adds a tool result message (role 'tool').
        function_name is optional but can be useful for context, though not directly sent in 'tool' role message usually.
        """
        content_str = json.dumps(result) if not isinstance(result, str) else result
        # The 'name' field is not typically used with role 'tool' in modern OpenAI API.
        # It was used with the legacy 'function' role for the result message.
        self.add_message(Message(role="tool", content=content_str, tool_call_id=tool_call_id, name=function_name))

    # Legacy support for function calling
    def add_function_call_decision_message(self, tool_calls: List[Dict]):
        """Adds an assistant message that specifically represents a function call decision via tool_calls."""
        self.add_assistant_message(tool_calls=tool_calls)

    def add_function_result_message(self, function_name: str, result: Any, tool_call_id: Optional[str] = None):
        """Adds a function result message (legacy role 'function' or modern 'tool').
           If tool_call_id is provided, it uses the 'tool' role. Otherwise, legacy 'function' role.
        """
        content_str = json.dumps(result) if not isinstance(result, str) else result
        if tool_call_id:
            self.add_message(Message(role="tool", content=content_str, tool_call_id=tool_call_id, name=function_name))
        else:
            # Legacy: role 'function' uses name for the function name.
            self.add_message(Message(role="function", content=content_str, name=function_name))

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the entire message history as a list of dictionaries for API calls."""
        return [msg.to_dict() for msg in self.messages]

    def get_last_n_messages(self, n: int, as_dicts: bool = True) -> List[Any]:
        """Returns the last N messages. If as_dicts is True, returns list of dicts, else list of Message objects."""
        last_n = self.messages[-n:]
        if as_dicts:
            return [msg.to_dict() for msg in last_n]
        return last_n

    def clear(self, keep_system_prompt: bool = True):
        """Clears the message history.
        If keep_system_prompt is True and a system prompt exists, it will be preserved.
        """
        if keep_system_prompt and self.messages and self.messages[0].role == "system":
            system_message = self.messages[0]
            self.messages = [system_message]
        else:
            self.messages = []

    def __repr__(self):
        return f"Memory(messages=['{', '.join(repr(m) for m in self.messages)}'])"

# Example Usage:
if __name__ == '__main__':
    # Test basic memory
    memory = Memory(system_prompt="You are a helpful assistant.")
    memory.add_user_message("Hello, who are you?")
    memory.add_assistant_message("I am an AI assistant. How can I help?")
    print("Basic Memory:")
    for msg_dict in memory.get_history():
        print(json.dumps(msg_dict, indent=2))
    print("---")

    # Test memory with tool calls (OpenAI current standard)
    tool_call_memory = Memory(system_prompt="You are a function-calling AI that uses tools.")
    tool_call_memory.add_user_message("What is the weather in London and what is 2+2?")
    
    # Simulate assistant deciding to call tools
    simulated_tool_calls = [
        {
            "id": "call_weather_123", "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"location": "London"})}
        },
        {
            "id": "call_calculator_456", "type": "function",
            "function": {"name": "calculate", "arguments": json.dumps({"expression": "2+2"})}
        }
    ]
    tool_call_memory.add_assistant_message(tool_calls=simulated_tool_calls)
    
    # Simulate tool execution and results
    tool_call_memory.add_tool_result_message(
        tool_call_id="call_weather_123", 
        function_name="get_weather", # name can be useful for logging/debugging, not always sent in 'tool' role
        result={"temperature": "15C", "condition": "Cloudy"}
    )
    tool_call_memory.add_tool_result_message(
        tool_call_id="call_calculator_456", 
        function_name="calculate",
        result={"result": 4}
    )
    
    tool_call_memory.add_assistant_message(content="The weather in London is 15C and cloudy. 2+2 equals 4.")

    print("Memory with Tool Calls:")
    for msg_dict in tool_call_memory.get_history():
        print(json.dumps(msg_dict, indent=2))
    print("---")

    # Test legacy function call memory
    legacy_func_memory = Memory(system_prompt="You are a legacy function-calling AI.")
    legacy_func_memory.add_user_message("What is the weather in Berlin?")
    
    # Simulate assistant deciding to call a function (legacy style, conceptualized as single call)
    # This is still stored as an assistant message with tool_calls if we map it to modern API
    # For a true legacy API, the format might be different (e.g. assistant message with function_call dict)
    # Our current Message class models this via assistant message + tool_calls
    legacy_tool_calls = [
        {"id": "legacy_call_789", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Berlin\", \"unit\": \"celsius\"}"}}
    ]
    legacy_func_memory.add_assistant_message(tool_calls=legacy_tool_calls) # Simulating decision

    # Simulate function execution and result (legacy style using role 'function')
    legacy_func_memory.add_function_result_message( # this will use role='function' if no tool_call_id
        function_name="get_current_weather", 
        result="{\"temperature\": \"12\", \"condition\": \"Sunny\"}" 
    )
    legacy_func_memory.add_assistant_message("The weather in Berlin is 12°C and Sunny.")
    print("Memory with Legacy Function Calls (using role 'function' for result):")
    for msg_dict in legacy_func_memory.get_history():
        print(json.dumps(msg_dict, indent=2))
    print("---")

    # Test clearing memory
    memory.clear(keep_system_prompt=True)
    print("Basic Memory after clearing (keeping system prompt):")
    print(json.dumps(memory.get_history(), indent=2))
    assert len(memory.messages) == 1
    assert memory.messages[0].role == "system"

    memory.clear(keep_system_prompt=False)
    print("Basic Memory after clearing (removing system prompt):")
    print(json.dumps(memory.get_history(), indent=2))
    assert len(memory.messages) == 0

    # Test message validation
    print("\nTesting Validations:")
    try:
        Message(role="assistant") # No content or tool_calls
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        Message(role="user", tool_calls=[{}]) # tool_calls on user message
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        Message(role="tool", content="result") # role tool without tool_call_id
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        Message(role="tool", tool_call_id="tid") # role tool without content
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        Message(role="function", content="res") # role function without name (legacy)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("Memory tests completed.") 