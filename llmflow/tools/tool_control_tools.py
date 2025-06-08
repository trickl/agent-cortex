"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Control Tools - Manages tool registration, activation, and coordination between different tools in the LLMFlow ecosystem.
"""

from typing import Dict, Any
from llmflow.tools.tool_decorator import register_tool

# @register_tool(tags=["system"], terminal=True)
# def terminate(message: str) -> str:
#     """
#     Terminate the agent's work with a final message.
#     This tool signals that all tasks are completed, a final result has been obtained,
#     or the agent has determined there's nothing more to do.
#     When this tool is called, the agent's processing cycle will stop, and the provided
#     message will be the final output.
#
#     Args:
#         message: The final message or result to be delivered. This should be informative.
#
#     Returns:
#         A confirmation string indicating the agent is terminating with the message.
#     """
#     # The actual termination logic will be handled by the agent based on the 'terminal=True' flag.
#     # This tool's return value is primarily for logging and confirmation if needed.
#     return f"Termination signal received. Final message: {message}"


@register_tool(tags=["system", "control"], terminal=True)
def conclude_current_turn(final_response: str) -> str:
    """
    Concludes the current processing turn for the user's query and provides the final response.
    Call this tool when all necessary actions (like searching, parsing, calculations) have been taken 
    and you have a complete and final answer ready for the user for their current query.
    The agent will stop its current internal iteration loop and present this response.

    Args:
        final_response: The complete and final response string to be given to the user.

    Returns:
        The final response string, which will be presented to the user.
    """
    # The 'terminal=True' flag ensures the agent's loop stops.
    # The string returned by this function becomes the agent's final message for this turn.
    return final_response 