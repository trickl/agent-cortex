"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Agent Module - The central orchestrator of the GAME methodology.
This module implements the main Agent class that coordinates all GAME components:
- Goals: Manages and tracks agent objectives
- Actions: Executes tools and handles responses
- Memory: Maintains conversation history and context
- Environment: Provides controlled access to external systems

Key Features:
- Flexible LLM integration through LLMClient
- Dynamic tool loading and filtering by tags
- Robust conversation management
- Iteration control and goal tracking
- Terminal tool support for graceful completion
- Comprehensive error handling and reporting
- Detailed logging in verbose mode

The Agent class serves as the primary interface for building AI agents
that can understand user requests, execute appropriate actions, and
maintain coherent conversations while working towards specific goals.
"""

import json
import time
from typing import List, Dict, Any, Optional

# LLM Client
from llmflow.llm_client import LLMClient # New LLM Client

# Core components
from .goals import GoalManager
from .memory import Memory # MessageRole (not directly used here, but Memory uses it)
from .action_handler import ActionHandler
from .environment import Environment

# Tooling
from llmflow.tools.tool_registry import get_all_tools_schemas, get_tools_by_tags

class Agent:
    """The AI Agent that uses Goals, Actions (Tools), Memory, and Environment."""

    _MAX_CONTEXT_TRACE = 100

    def __init__(
        self,
        llm_client: LLMClient,  # Changed from llm_provider
        system_prompt: str = "You are a helpful AI assistant. You have access to tools. Be clear about tool usage. When a user asks for information that might require searching the web (e.g., news, facts, etc.): \\\\n+1. First, use a search tool to find relevant URLs and their snippets. \\\\n+2. Then, analyze the search results. If appropriate, use a web parsing tool to fetch content from one or more of the most relevant URLs. \\\\n+3. Finally, synthesize the gathered information and provide a comprehensive answer or summary to the user. If you parse content, mention the source URLs.",
        initial_goals: Optional[List[Dict[str, Any]]] = None,
        available_tool_tags: Optional[List[str]] = None,
        match_all_tags: bool = True,
        max_iterations: Optional[int] = 10,  # Allow None for unlimited
        verbose: bool = True,
        llm_max_retries: int = 2,
        llm_retry_delay: float = 2.0,
    ):
        self.llm_client = llm_client # Changed from llm_provider
        self.goal_manager = GoalManager(initial_goals=initial_goals)
        self.memory = Memory(system_prompt=system_prompt)
        self.environment = Environment()
        self.action_handler = ActionHandler(environment_execute_action_callback=self.environment.execute_action)
        
        self.available_tool_tags = available_tool_tags
        self.match_all_tags_for_tools = match_all_tags
        self.verbose = verbose
        self.active_tools_schemas: List[Dict[str, Any]] = []
        self.context_trace: List[Dict[str, Any]] = []
        self._load_active_tools()

        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.llm_max_retries = max(0, llm_max_retries)
        self.llm_retry_delay = max(0.0, llm_retry_delay)

        if self.verbose:
            print("Agent initialized with LLMClient.")
            print(f"System Prompt: {system_prompt}")
            if self.max_iterations is None:
                print("Max Iterations: Unlimited")
            else:
                print(f"Max Iterations: {self.max_iterations}")
            self.goal_manager.get_goals_for_prompt()
            print(f"Loaded {len(self.active_tools_schemas)} tools for the agent.")
            print(
                f"LLM retry policy: {self.llm_max_retries} retries with base delay {self.llm_retry_delay:.1f}s."
            )

    def _load_active_tools(self):
        """Loads tool schemas based on specified tags."""
        if self.available_tool_tags:
            tagged_tools = get_tools_by_tags(tags=self.available_tool_tags, match_all=self.match_all_tags_for_tools)
            self.active_tools_schemas = [info["schema"] for info in tagged_tools.values()]
        else:
            self.active_tools_schemas = get_all_tools_schemas()
        
        if self.verbose and not self.available_tool_tags:
            print("No specific tool tags provided, agent has access to all registered tools.")
        elif self.verbose:
            print(f"Agent configured with tools matching tags: {self.available_tool_tags} (match_all: {self.match_all_tags_for_tools})")

    def _construct_prompt(self) -> List[Dict[str, Any]]:
        """Constructs the full prompt for the LLM, including history and goals."""
        messages = self.memory.get_history() # Get a mutable copy
        current_goals_text = self.goal_manager.get_goals_for_prompt()
        
        user_message_found_and_updated = False
        if messages:
            # Try to append goals to the last actual user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Ensure 'content' exists and is a string
                    if "content" not in messages[i] or messages[i]["content"] is None:
                        messages[i]["content"] = "" # Initialize if null or missing
                    elif not isinstance(messages[i]["content"], str):
                         # If content is not a string (e.g. list of content blocks for multimodal),
                         # we might need a more sophisticated way to append goals.
                         # For now, let's convert it to string and append, or just append a new message.
                         # This part might need refinement based on how complex message structures are handled.
                         # For simplicity, if it's not a simple string, we'll fall back to adding a new message.
                         print(f"Warning: Last user message content is not a simple string: {type(messages[i]['content'])}. Appending goals as new message.")
                         break # Break to fall through to the new message logic

                    messages[i]["content"] += f"\\n\\n--- Current Goals ---\\n{current_goals_text}\\n--- End of Goals ---"
                    user_message_found_and_updated = True
                    break
        
        if not user_message_found_and_updated:
            # If no user message was found (e.g., history is empty or only system/assistant messages)
            # or if the last user message content was complex,
            # add goals as a new user-style message.
            messages.append({"role": "user", "content": f"--- Review Current Goals ---\\n{current_goals_text}\\n--- End of Goals ---"})
            
        return messages

    def _record_context_snapshot(
        self,
        stage: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        snapshot = messages if messages is not None else self.memory.get_history()
        msg_count = len(snapshot)
        approx_chars = sum(
            len(str(msg.get("content", "")))
            for msg in snapshot
            if isinstance(msg, dict)
        )
        tool_messages = sum(
            1 for msg in snapshot if isinstance(msg, dict) and msg.get("role") == "tool"
        )
        entry = {
            "stage": stage,
            "message_count": msg_count,
            "approx_chars": approx_chars,
            "tool_messages": tool_messages,
            "timestamp": time.time(),
        }
        self.context_trace.append(entry)
        if len(self.context_trace) > self._MAX_CONTEXT_TRACE:
            self.context_trace.pop(0)
        if self.verbose:
            print(
                f"[context] {stage}: {msg_count} msgs, ~{approx_chars} chars, {tool_messages} tool msgs."
            )

    def get_context_trace(self) -> List[Dict[str, Any]]:
        """Return a copy of the recent context trace for diagnostics."""

        return list(self.context_trace)

    def _call_llm_with_retries(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        attempts = 0
        last_error: Optional[str] = None
        max_attempts = self.llm_max_retries + 1

        while attempts < max_attempts:
            try:
                response = self.llm_client.generate(messages=messages, tools=tools)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = f"LLM invocation raised {exc.__class__.__name__}: {exc}"
            else:
                if isinstance(response, dict):
                    content = response.get("content")
                    if (
                        response.get("role") == "assistant"
                        and isinstance(content, str)
                        and content.strip().startswith("Error:")
                    ):
                        last_error = content
                    else:
                        return response
                else:
                    last_error = f"Unexpected LLM response type: {type(response)}"

            attempts += 1
            if attempts < max_attempts:
                delay = self.llm_retry_delay * max(attempts, 1)
                if self.verbose:
                    print(
                        f"LLM call failed (attempt {attempts}/{self.llm_max_retries}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                if delay > 0:
                    time.sleep(delay)

        raise RuntimeError(last_error or "LLM call failed without additional details.")

    def _summarize_tool_failure(
        self,
        tool_name: Optional[str],
        metadata: Dict[str, Any],
        fallback_content: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        name = tool_name or "unknown_tool"
        severity = "fatal" if metadata.get("fatal") else ("retryable" if metadata.get("retryable", False) else "non-retryable")
        reason = metadata.get("error") or (fallback_content[:200] if fallback_content else "Unknown error")
        note = (
            f"[observation] Tool '{name}' reported a {severity} failure: {reason}."
        )

        if metadata.get("fatal"):
            note += " Terminating the current run."
        elif metadata.get("retryable", False):
            note += " The error may be transient; consider retrying with adjusted parameters."
        else:
            note += " Please adjust the plan, arguments, or choose a different tool before continuing."

        user_message = None
        if metadata.get("fatal"):
            user_message = f"Unable to continue because tool '{name}' encountered a fatal error: {reason}."

        return {"note": note, "user_message": user_message}

    def _check_completion(self) -> bool:
        """Checks if the agent's goals are completed or max iterations reached."""
        if self.goal_manager.all_goals_completed():
            if self.verbose: print("All goals completed.")
            return True
        if self.max_iterations is not None and self.max_iterations > 0 and self.current_iteration >= self.max_iterations:
            if self.verbose: print("Max iterations reached.")
            return True
        return False

    def add_user_message_and_run(self, user_input: str) -> Optional[str]:
        """Adds a user message and runs the agent loop until it needs new input or finishes.
        Returns the final agent message content as a string, or None if no specific message was produced.
        """
        self.memory.add_user_message(user_input)
        if self.verbose: print(f"\\nIteration {self.current_iteration + 1}: User says: '{user_input}'")
        
        # Reset current_iteration for a new user message if agent is not designed to be strictly goal-completion oriented
        # For an interactive chat, each user message starts a new "turn" or mini-session.
        # If max_iterations is meant per user message, reset it here.
        # If agent is strictly goal-oriented and continues across user messages until goals are met, don't reset.
        # For this interactive loop, resetting seems more appropriate.
        self.current_iteration = 0 
        
        final_agent_message_content = self.run_main_loop() # run_main_loop will now return the string content
        return final_agent_message_content

    def run_main_loop(self) -> Optional[str]:
        """Runs the main agent cycle: Prompt -> LLM -> Parse -> Execute -> Update Memory -> Check Completion.
        Returns the final agent message string for the user, or None.
        """
        
        should_terminate_agent = False
        final_agent_message = None

        while not self._check_completion() and not should_terminate_agent:
            self.current_iteration += 1
            if self.verbose:
                if self.max_iterations is not None and self.max_iterations > 0:
                    print(f"\\n--- Agent Iteration: {self.current_iteration}/{self.max_iterations} ---")
                else:
                    print(f"\\n--- Agent Iteration: {self.current_iteration} ---")

            llm_messages = self._construct_prompt()
            if self.verbose:
                print("Constructed LLM Messages:")
                for msg in llm_messages[-3:]: # Show last few for brevity
                    content_summary = str(msg.get('content', ''))[:100].replace('\\n', ' ') + "..."
                    tool_calls_summary = msg.get('tool_calls') if msg.get('tool_calls') else ''
                    print(f"  {msg['role']}: {content_summary} {tool_calls_summary}")
                approx_chars = sum(len(str(msg.get('content', ''))) for msg in llm_messages if isinstance(msg, dict))
                print(f"Context stats: {len(llm_messages)} messages, ~{approx_chars} content characters.")

            self._record_context_snapshot("pre_llm_prompt", llm_messages)

            if self.verbose: print("Querying LLM via LLMClient...")
            
            try:
                llm_response = self._call_llm_with_retries(
                    llm_messages,
                    self.active_tools_schemas if self.active_tools_schemas else None,
                )
            except RuntimeError as exc:
                final_agent_message = (
                    "Unable to continue because the language model request failed repeatedly: "
                    f"{exc}"
                )
                print(final_agent_message)
                self.memory.add_assistant_message(content=final_agent_message)
                break
            
            # The new LLMClient directly returns the assistant message object (or an error structure)
            assistant_message_obj = llm_response 

            if not isinstance(assistant_message_obj, dict) or \
               assistant_message_obj.get("role") != "assistant" or \
               (assistant_message_obj.get("content") is None and assistant_message_obj.get("tool_calls") is None and "Error:" not in str(assistant_message_obj.get("content")) ):
                error_content = f"Error: LLM response format is not as expected or indicates an error. Received: {assistant_message_obj}"
                print(error_content)
                self.memory.add_assistant_message(content=error_content)
                break 

            if "Error:" in str(assistant_message_obj.get("content")) and assistant_message_obj.get("role") == "assistant":
                error_text = assistant_message_obj.get("content")
                print(f"LLMClient reported an error: {error_text}")
                self.memory.add_assistant_message(content=error_text, tool_calls=None)
                final_agent_message = error_text
                break  # Stop processing this turn if LLM had an error

            self.memory.add_assistant_message(content=assistant_message_obj.get("content"),
                                            tool_calls=assistant_message_obj.get("tool_calls"))
            
            if self.verbose:
                print("LLM Response Received (via LLMClient):")
                print(f"  Role: {assistant_message_obj.get('role')}")
                if assistant_message_obj.get("content"):
                    print(f"  Content: {assistant_message_obj.get('content')}")
                if assistant_message_obj.get("tool_calls"):
                    print(f"  Tool Calls: {json.dumps(assistant_message_obj.get('tool_calls'), indent=2)}")

            # Action parsing expects a structure like {"message": assistant_message_obj}
            # if the assistant_message_obj is the direct message part.
            # UPDATE: The LLMClient already returns the assistant message object directly.
            # So, we pass it directly to the parser, which expects to find 'tool_calls' in it.
            action_requests = self.action_handler.parse_llm_response_for_tool_calls(assistant_message_obj)

            if action_requests:
                if self.verbose: print(f"Identified {len(action_requests)} action(s) to execute.")
                tool_results = self.action_handler.execute_tool_calls(action_requests)
                
                if self.verbose: print("Adding tool results to memory...")
                for res_dict in tool_results:
                    if res_dict.get("role") == "tool":
                        self.memory.add_tool_result_message(
                            tool_call_id=res_dict["tool_call_id"],
                            result=res_dict.get("content"),
                            function_name=res_dict.get("name")
                        )
                        if self.verbose: print(f"  Added tool result: {res_dict.get('name') or res_dict['tool_call_id']} -> {str(res_dict['content'])[:100]}...")

                        metadata = res_dict.get("metadata") or {}
                        tool_failed = not metadata.get("success", True)
                        if tool_failed:
                            summary = self._summarize_tool_failure(
                                res_dict.get("name"), metadata, res_dict.get("content")
                            )
                            note = summary.get("note")
                            if note:
                                if self.verbose:
                                    print(f"  Controller note: {note}")
                                self.memory.add_assistant_message(content=note)
                            if metadata.get("fatal"):
                                should_terminate_agent = True
                                final_agent_message = summary.get("user_message") or note
                                break
                        
                        # Check for terminal tool execution
                        if res_dict.get("is_terminal", False):
                            should_terminate_agent = True
                            final_agent_message = res_dict.get("content") 
                            if self.verbose:
                                print(f"Terminal tool '{res_dict.get('name')}' executed. Agent will stop.")
                                print(f"Final message from terminal tool: {final_agent_message}")
                            # If one terminal tool is found, we typically stop processing further tools in this batch,
                            # or let all tools in the batch run and then terminate.
                            # For simplicity, we'll let all tools in the current batch run, but the agent will terminate afterwards.
                            # If multiple terminal tools are called, the last one's message might be captured, or the first.
                            # Current logic: first terminal tool encountered will set the final message.
                            # To capture last: move final_agent_message assignment outside the loop, checking should_terminate_agent.
                            # For now, first one wins for the message, but agent terminates after this iteration regardless.
                            # No, let's ensure the final_agent_message is set. If it's a JSON string from the tool, parse it.
                            try:
                                # The `terminate` tool returns a string like "Termination signal received. Final message: ..."
                                # We might want to extract the actual user-facing message from it if it's structured.
                                # For now, the raw content from the terminal tool is the final message.
                                # If the tool output is JSON with an error, we should show that too.
                                parsed_content = json.loads(final_agent_message)
                                if isinstance(parsed_content, dict) and "error" in parsed_content:
                                    final_agent_message = f"Terminating due to tool error: {parsed_content.get('details', str(parsed_content))}"
                                elif isinstance(parsed_content, dict) and "message" in parsed_content: # If tool returns e.g. {"message": "final text"}
                                    final_agent_message = parsed_content["message"]
                                # If it's the specific format from our terminate tool: "Termination signal received. Final message: ACTUAL_MESSAGE"
                                elif isinstance(final_agent_message, str) and final_agent_message.startswith("Termination signal received. Final message: "):
                                    final_agent_message = final_agent_message.replace("Termination signal received. Final message: ", "", 1)

                            except json.JSONDecodeError:
                                # Not JSON, use as is. This is expected for the terminate tool.
                                pass 
                            # Break from iterating over tool results if a terminal tool is found, agent will stop after this iteration.
                            break # Stop processing other tools in this turn if one is terminal

                    else:
                        if self.verbose: print(f"Warning: Skipping unexpected message type from tool results: {res_dict}")
                
                if should_terminate_agent:
                    break # Break from the main while loop

                self._record_context_snapshot("post_iteration_state")
            else:
                if self.verbose: print("No tool calls from LLM. Agent response is the content.")
                # If LLM provided content, that's the response for this iteration.
                # If the agent is not meant to continue after a direct textual response, this `break` is correct.
                final_agent_message = assistant_message_obj.get("content") # Capture LLM direct response as potential final message if no tools were called
                self._record_context_snapshot("post_iteration_state")
                break 

        if self.verbose:
            if should_terminate_agent:
                print(f"Agent finished: Terminal tool executed. Final message: {final_agent_message}")
                # Add the final message from the terminal tool (or LLM if no tool call) as the last assistant message if not already there
                # Check if the last message in memory is already this final message
                last_mem_msg = self.memory.get_last_n_messages(1, as_dicts=True)
                if not (last_mem_msg and last_mem_msg[0]["role"] == "assistant" and last_mem_msg[0]["content"] == final_agent_message):
                     # If the terminal tool's output wasn't an assistant message already added
                     # (it was a 'tool' role message), add a final assistant message.
                     # Or if the final message came from LLM directly.
                    self.memory.add_assistant_message(content=final_agent_message)

            elif self.goal_manager.all_goals_completed():
                print("Agent finished: All goals completed.")
            elif self.max_iterations is not None and self.current_iteration >= self.max_iterations:
                print("Agent finished: Max iterations reached.")
            else:
                print("Agent loop ended. Awaiting next user input or instruction.")
        
        # Instead of returning self.memory.get_history(), return the specific final message for the user.
        return final_agent_message

# --- Example Usage ---
# if __name__ == '__main__':
#     # Ensure some tools are registered for the agent to use.
#     # This typically happens when tool definition files are imported.
#     # For this example, let's try to import from tool_decorator.py which has example tools.
#     # try:
#     #     from llmflow.tools.tool_decorator import read_file, write_to_file, get_user_name, process_data, _tool_registry
#     #     if not _tool_registry: # If tool_decorator.py didn't run its main block to register tools
#     #         print("Agent Example: Manually registering dummy tools as registry from tool_decorator is empty.")
#     #         # Define some dummy tools directly for the agent to use if not loaded
#     #         from llmflow.tools.tool_decorator import register_tool # Import for direct use
#     #         @register_tool(tags=["weather"])
#     #         def get_weather(location: str, unit: Optional[str] = "celsius") -> Dict[str, str]:
#     #             """Gets the current weather for a location.""" 
#     #             print(f"[Dummy Tool] get_weather called for {location} in {unit}")
#     #             if location.lower() == "simcity":
#     #                 return {"location": location, "temperature": f"22 {unit}", "condition": "Sunny simulation"}
#     #             return {"location": location, "temperature": "unknown", "condition": "unknown"}
#     #         
#     #         @register_tool(tags=["info"])
#     #         def get_user_details(user_id: str) -> Dict[str, str]:
#     #             """Retrieves details for a given user ID."""
#     #             return {"user_id": user_id, "name": "Simulated User", "status": "active"}
#     #
#     # except ImportError as e:
#     #     print(f"Could not import example tools: {e}. Agent might have no tools.")
# 
#     # The tools from tool_decorator.py should now be registered automatically upon import.
#     # We can check tool_registry to confirm.
#     from llmflow.tools.tool_registry import list_available_tools
#     print("Verifying registered tools before agent initialization:")
#     available_tool_names = list_available_tools(verbose=False)
#     if not available_tool_names:
#         print("Warning: No tools seem to be registered from tool_decorator.py or other tool modules.")
#     else:
#         print(f"  {len(available_tool_names)} tools available: {available_tool_names}")
# 
#     print("\\n--- AGENT SIMULATION START ---")
#     
#     # Initialize the LLM Provider (using our mock one)
#     mock_llm = OpenAILikeLLM(model_name="simulated-gpt-3.5")
# 
#     # Define initial goals for the agent
#     agent_goals = [
#         {"description": "Greet the user and ask how I can help.", "priority": 1},
#         {"description": "If the user asks for weather in SimCity, provide it.", "priority": 0}
#     ]
# 
#     # Initialize the Agent
#     # Agent will only use tools tagged with "weather" or all tools if available_tool_tags=None
#     ai_agent = Agent(
#         llm_client=mock_llm, 
#         system_prompt="You are a friendly and helpful assistant that can use tools.",
#         initial_goals=agent_goals,
#         # available_tool_tags=["weather"], # Uncomment to restrict to only weather tools
#         available_tool_tags=None, # Agent has access to all registered tools
#         max_iterations=5, 
#         verbose=True
#     )
# 
#     # --- Interaction 1: User says hello ---
#     print("\\n--- Interaction 1 ---")
#     conversation_history = ai_agent.add_user_message_and_run("Hello agent!")
#     # print("\\nFinal Conversation History (Interaction 1):")
#     # for msg in conversation_history:
#     #     print(msg)
#     
#     # --- Interaction 2: User asks to read a file (should trigger read_file tool call) ---
#     if not ai_agent._check_completion(): # Continue if agent hasn't finished
#         print("\\n--- Interaction 2 --- ")
#         conversation_history = ai_agent.add_user_message_and_run("Can you read file important_document.txt for me?")
#         # print("\\nFinal Conversation History (Interaction 2):")
#         # for msg in conversation_history:
#         #     print(msg)
#     
#     # --- Interaction 3: User asks to add numbers (should trigger add_numbers tool call) ---
#     if not ai_agent._check_completion(): # Continue if agent hasn't finished
#         print("\\n--- Interaction 3 --- ")
#         conversation_history = ai_agent.add_user_message_and_run("Please add 123 and 456.")
#         # print("\\nFinal Conversation History (Interaction 3):")
#         # for msg in conversation_history:
#         # print(msg)
# 
#     # --- Interaction 4: Potentially ask about user details if that tool is available ---
#     # Note: get_user_details is not currently among the registered tools from tool_decorator.py
#     # We would need to add it there for this to work as intended.
#     # if not ai_agent._check_completion() and any(t["function"]["name"] == "get_user_details" for t in ai_agent.active_tools_schemas):
#     #     print("\\n--- Interaction 4 --- ")
#     #     conversation_history = ai_agent.add_user_message_and_run("Can you get details for user_id 'sim123'?")
#     
#     print("\\n--- AGENT SIMULATION END ---")
#     print("Final Goals Status:")
#     print(ai_agent.goal_manager.get_goals_for_prompt())
#     print("\\nFinal Memory (last few messages):")
#     for msg_dict in ai_agent.memory.get_last_n_messages(5, as_dicts=True):
#         print(f"  {msg_dict.get('role')}: {str(msg_dict.get('content',''))[:100]}... {msg_dict.get('tool_calls') if msg_dict.get('tool_calls') else ''}") 