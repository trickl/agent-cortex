"""
Main script to run a simulation of the agent framework.
"""
import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the project root to sys.path to allow for absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llmflow.core.agent import Agent
from llmflow.llm_client import LLMClient
import llmflow.tools # To ensure tools are registered
from llmflow.tools.tool_registry import list_available_tools, get_all_tools_schemas

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_simulation_config(file_path: str = "simulation_config.json") -> Dict[str, Any]:
    """Loads the simulation configuration from a JSON file."""
    if not os.path.exists(file_path):
        logger.warning(f"Simulation config file '{file_path}' not found. Using default interaction.")
        # Default interaction if no config file
        return {
            "agent_system_prompt": "You are a friendly and helpful assistant named SIM-Agent. You can use tools to help the user. Always inform the user about the tool actions you take.",
            "initial_agent_goals": [
                {"description": "Greet the user and ask how I can help.", "priority": 1},
                {"description": "If the user asks to read a file, read it using the 'read_file' tool.", "priority": 0},
                {"description": "If the user asks to add numbers, use the 'add_numbers' tool.", "priority": 0},
                {"description": "If the user asks for a web search, use the 'duckduckgo_search' tool.", "priority": 0}
            ],
            "simulation_interactions": [
                {"user_input": "Hello agent!"},
                {"user_input": "Can you read file example.txt for me?"},
                {"user_input": "Thanks! Now, please add 1234 and 5678."},
                {"user_input": "What are the latest news about AI?"}
            ],
            "llm_model_name": "gpt-4o-mini",
            "max_agent_iterations_per_turn": 6,
            "available_tool_tags": None, # None means all tools are available
            "match_all_tags": True
        }
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading simulation config '{file_path}': {e}", exc_info=True)
        sys.exit(1)

def run_agent_simulation():
    """Runs a predefined agent simulation based on a configuration file."""
    load_dotenv() # Load .env for API keys
    config = load_simulation_config()

    print("--- AGENT SIMULATION START ---")
    
    model_name = config.get("llm_model_name", "gpt-4o-mini")
    print(f"Using LLM Model: {model_name}")
    try:
        llm_client = LLMClient() # Reads from llm_config.yaml by default
        logger.info(f"LLMClient initialized using provider and model from llm_config.yaml. CLI model '{model_name}' from sim config noted.")
    except Exception as e:
        logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
        print(f"Error: Could not initialize LLMClient. Details: {e}")
        return

    print("\nVerifying registered tools before agent initialization:")
    tool_names = list_available_tools(verbose=False)
    print(f"  {len(tool_names)} tools available: {tool_names}")
    # print("\nTool Schemas:")
    # for schema in get_all_tools_schemas():
    #     print(json.dumps(schema, indent=2))

    print("Environment initialized.")

    agent_system_prompt = config.get("agent_system_prompt", "You are a helpful AI assistant.")
    agent_goals = config.get("initial_agent_goals", [])
    available_tags = config.get("available_tool_tags", None)
    match_all = config.get("match_all_tags", True)
    max_iterations = config.get("max_agent_iterations_per_turn", 0) # 0 means None/unlimited for Agent

    print(f"No specific tool tags provided, agent has access to all registered tools." if not available_tags else f"Agent tools filtered by tags: {available_tags}")

    try:
        agent_settings = {
            "llm_client": llm_client, 
            "system_prompt": agent_system_prompt,
            "initial_goals": agent_goals,
            "available_tool_tags": available_tags,
            "match_all_tags": match_all,
            "verbose": True
        }
        if max_iterations > 0:
            agent_settings["max_iterations"] = max_iterations
            print(f"Max Iterations: {max_iterations}")
        else:
            agent_settings["max_iterations"] = None
            print("Max Iterations: Unlimited")
            
        ai_agent = Agent(**agent_settings)
        logger.info(f"Loaded {len(ai_agent.active_tools_schemas)} tools for the agent.")

    except Exception as e:
        logger.error(f"Failed to initialize Agent: {e}", exc_info=True)
        print(f"Error: Could not initialize Agent. Details: {e}")
        return

    interactions = config.get("simulation_interactions", [])
    for i, interaction in enumerate(interactions):
        user_input = interaction.get("user_input")
        if user_input is None:
            logger.warning(f"Skipping interaction {i+1} due to missing 'user_input'.")
            continue
        
        print(f"\n--- Interaction {i+1} --- ")
        # Agent's add_user_message_and_run now returns the final message string
        final_message = ai_agent.add_user_message_and_run(user_input)
        print(f"\nAssistant: {final_message}")
        
        if ai_agent._check_completion(): # Check if agent decided to stop or completed goals
            logger.info("Agent indicated completion or max iterations reached after this interaction.")
            # break # Optionally break simulation if agent completes all goals early

    print("\n--- AGENT SIMULATION END ---")
    print("Final Goals Status:")
    print(ai_agent.goal_manager.get_goals_for_prompt())
    print("\nFinal Memory (last 5 messages):")
    for msg_dict in ai_agent.memory.get_last_n_messages(5, as_dicts=True):
        role = msg_dict.get('role')
        content = str(msg_dict.get('content',''))[:100] + ("..." if len(str(msg_dict.get('content',''))) > 100 else "")
        tool_calls = msg_dict.get('tool_calls')
        print(f"  {role}: {content} {json.dumps(tool_calls) if tool_calls else ''}")

if __name__ == "__main__":
    run_agent_simulation() 