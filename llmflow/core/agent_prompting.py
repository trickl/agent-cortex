"""Legacy prompt helper removed in favor of the Java plan agent."""

raise RuntimeError(
    "llmflow.core.agent_prompting has been retired; prompts now live inside the Java-plan Agent."
)

import time
from typing import Any, Dict, List, Optional


class AgentPromptMixin:
    """Provides prompt construction and LLM wrapper utilities."""

    def _pending_goal_count(self) -> int:
        return sum(1 for goal in self.goal_manager.goals if not goal.completed)

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

                    messages[i]["content"] += f"\n\n--- Current Goals ---\n{current_goals_text}\n--- End of Goals ---"
                    user_message_found_and_updated = True
                    break
        
        if not user_message_found_and_updated:
            # If no user message was found (e.g., history is empty or only system/assistant messages)
            # or if the last user message content was complex,
            # add goals as a new user-style message.
            messages.append({"role": "user", "content": f"--- Review Current Goals ---\n{current_goals_text}\n--- End of Goals ---"})
            
        return messages

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

        retryable = metadata.get("retryable", False)
        fatal = metadata.get("fatal")

        if fatal:
            note += " Terminating the current run."
        elif retryable:
            note += " The error may be transient; consider retrying with adjusted parameters."
        else:
            note += " Please adjust the plan, arguments, or choose a different tool before continuing."

        user_message = None
        if fatal:
            user_message = (
                f"Unable to continue because tool '{name}' encountered a fatal error: {reason}."
            )
        elif not retryable:
            user_message = (
                f"Stopping because tool '{name}' reported a non-retryable error: {reason}."
            )

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
