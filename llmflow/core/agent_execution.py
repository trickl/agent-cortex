"""Legacy execution mixin removed in favor of CPL planning."""

raise RuntimeError(
    "llmflow.core.agent_execution is no longer supported; use the CPL-based Agent implementation."
)

import json
from typing import Dict, List, Optional

from .agent_tool_utils import _prepare_tool_result_content


class AgentExecutionMixin:
    """Provides the main run loop and user entrypoint."""

    def add_user_message_and_run(self, user_input: str) -> Optional[str]:
        """Adds a user message and runs the agent loop until it needs new input or finishes.
        Returns the final agent message content as a string, or None if no specific message was produced.
        """
        self._start_run_instrumentation()
        try:
            self.memory.add_user_message(user_input)
            if self.verbose: print(f"\nIteration {self.current_iteration + 1}: User says: '{user_input}'")
            
            # Reset current_iteration for a new user message if agent is not designed to be strictly goal-completion oriented
            # For an interactive chat, each user message starts a new "turn" or mini-session.
            # If max_iterations is meant per user message, reset it here.
            # If agent is strictly goal-oriented and continues across user messages until goals are met, don't reset.
            # For this interactive loop, resetting seems more appropriate.
            self.current_iteration = 0 
            
            final_agent_message_content = self.run_main_loop() # run_main_loop will now return the string content
            return final_agent_message_content
        finally:
            self._finalize_run_instrumentation()

    def run_main_loop(self) -> Optional[str]:
        """Runs the main agent cycle: Prompt -> LLM -> Parse -> Execute -> Update Memory -> Check Completion.
        Returns the final agent message string for the user, or None.
        """
        
        should_terminate_agent = False
        final_agent_message = None

        while not self._check_completion() and not should_terminate_agent:
            self.current_iteration += 1
            if self._mermaid_recorder:
                self._mermaid_recorder.record_plan_attempt(
                    self.current_iteration,
                    "start",
                    f"pending_goals={self._pending_goal_count()}",
                )
            if self.verbose:
                if self.max_iterations is not None and self.max_iterations > 0:
                    print(f"\n--- Agent Iteration: {self.current_iteration}/{self.max_iterations} ---")
                else:
                    print(f"\n--- Agent Iteration: {self.current_iteration} ---")

            llm_messages = self._construct_prompt()
            self._log_llm_prompt(llm_messages)
            if self.verbose:
                print("Constructed LLM Messages:")
                for msg in llm_messages[-3:]: # Show last few for brevity
                    content_summary = str(msg.get('content', ''))[:100].replace('\n', ' ') + "..."
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
                self._mark_run_failure()
                break
            
            # The new LLMClient directly returns the assistant message object (or an error structure)
            assistant_message_obj = llm_response 

            if not isinstance(assistant_message_obj, dict) or \
               assistant_message_obj.get("role") != "assistant" or \
               (assistant_message_obj.get("content") is None and assistant_message_obj.get("tool_calls") is None and "Error:" not in str(assistant_message_obj.get("content")) ):
                error_content = f"Error: LLM response format is not as expected or indicates an error. Received: {assistant_message_obj}"
                print(error_content)
                self.memory.add_assistant_message(content=error_content)
                self._mark_run_failure()
                break 

            if "Error:" in str(assistant_message_obj.get("content")) and assistant_message_obj.get("role") == "assistant":
                error_text = assistant_message_obj.get("content")
                print(f"LLMClient reported an error: {error_text}")
                self.memory.add_assistant_message(content=error_text, tool_calls=None)
                final_agent_message = error_text
                self._mark_run_failure()
                break  # Stop processing this turn if LLM had an error

            self.memory.add_assistant_message(content=assistant_message_obj.get("content"),
                                            tool_calls=assistant_message_obj.get("tool_calls"))
            self._log_llm_response(assistant_message_obj)
            
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
                        raw_result_content = res_dict.get("content")
                        result_content = raw_result_content
                        if isinstance(result_content, str):
                            result_content = _prepare_tool_result_content(
                                res_dict.get("name"), result_content
                            )
                        self.memory.add_tool_result_message(
                            tool_call_id=res_dict["tool_call_id"],
                            result=result_content,
                            function_name=res_dict.get("name")
                        )
                        if self.verbose: print(f"  Added tool result: {res_dict.get('name') or res_dict['tool_call_id']} -> {str(res_dict['content'])[:100]}...")

                        metadata = res_dict.get("metadata") or {}
                        self._log_tool_result(res_dict.get("name"), metadata, raw_result_content if isinstance(raw_result_content, str) else str(raw_result_content))
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
                            fatal = metadata.get("fatal")
                            retryable = metadata.get("retryable", False)
                            if fatal or not retryable:
                                self._mark_run_failure()
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
