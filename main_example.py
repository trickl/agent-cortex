import sys
import os
import locale
import argparse
import logging
from dotenv import load_dotenv

from llmflow.core.agent import Agent
from llmflow.llm_client import LLMClient
import llmflow.tools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stdin.encoding != 'utf-8':
    sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='utf-8', buffering=1)

current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

def run_interactive_chat(model_name: str = "gpt-4", max_iterations: int = 0):
    """Runs an interactive chat session with the AI agent."""
    load_dotenv()

    print(f"Using LLM Model: {model_name}")
    print("Initializing LLMClient...")
    try:
        llm_client = LLMClient()
        logger.info(f"LLMClient initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
        print(f"Error: Could not initialize LLMClient. Please check your API keys and configurations. Details: {e}")
        return

    print("Environment initialized.")

    system_prompt = """You are a powerful AI assistant with access to a wide range of tools and capabilities.
When users ask for information that requires web search (like news, facts, current events):
1. ALWAYS use the duckduckgo_search tool to find relevant information
2. Analyze the search results and synthesize a response
3. Include sources and relevant quotes in your answer
4. If needed, use web_parse_url tool to get more detailed content from specific URLs


Your available tools include:

1. File Operations: Read, write, and manage files and directories
2. Web Interaction: Search the web, parse web content, and process URLs
3. System Tools: Monitor system resources, execute shell commands
4. Cloud Operations: Interact with cloud services and APIs
5. Communication: Send emails, messages, and manage calendars
6. Data Processing: Analyze data, work with SQL databases, perform mathematical operations
7. Media Processing: Handle text-to-speech, speech recognition (using the 'transcribe_file' tool), image processing, and video downloading/processing
   - You can download videos from YouTube using the 'download_youtube_video' tool
   - Example: To download a video, use the tool with the YouTube URL
8. Code Execution: Run and analyze code in various languages
   - **IMPORTANT:** Do NOT use `execute_background_command` or direct calls to external programs like `ffmpeg` or `whisper` for tasks that have dedicated tools (e.g., transcription). Always use the appropriate specialized tool (`transcribe_file` for speech recognition).

When using these tools:
1. Choose the most appropriate tool for each task
2. Chain tools together when needed for complex operations
3. Handle errors gracefully and provide clear feedback
4. Always verify the results of tool operations
5. Protect sensitive information and follow security best practices

For YouTube video downloads:
1. **ALWAYS** use the 'download_youtube_video' tool when asked to download a YouTube video.
2. **NEVER** use Google Drive or other cloud download tools for YouTube URLs.
3. The 'download_youtube_video' tool requires the YouTube video URL as input.
4. The video will be saved to the default downloads directory.
5. Always confirm to the user when the download is complete.

For web-related queries:
1. Use search tools to find relevant information
2. Parse and analyze web content when needed
3. Synthesize information from multiple sources
4. Cite sources when providing information
5. Call 'conclude_current_turn' with your final response

For YouTube video download and transcription requests:
1. When asked to download and transcribe a YouTube video, first use the 'download_youtube_video' tool with the provided URL.
2. Once the download is successful, note the exact 'filename' (including the full path) result from the 'download_youtube_video' tool.
3. If any intermediate audio or video processing is required (e.g., format conversion), use the appropriate tool and ensure you keep track of the resulting exact file path.
4. **Crucially, after the download and any necessary processing, you MUST use the 'transcribe_file' tool for the transcription.**
5. Provide the **exact** path to the final processed audio/video file (obtained from step 2 or 3) as the 'file_path' argument to the 'transcribe_file' tool. You should also specify the 'language' (e.g., 'ru') if known, and an appropriate 'output_filename' for the transcription result.
6. The task of download AND transcription is complete **ONLY** when the 'transcribe_file' tool successfully produces the transcription output. Report the transcription content or the path to the transcription file to the user.
7. **DO NOT** conclude the turn or move to another task before the transcription is successfully generated by `transcribe_file`.
"""

    print("Initializing Agent...")
    try:
        agent_settings = {
            "llm_client": llm_client,
            "system_prompt": system_prompt,
            "initial_goals": None,
            "available_tool_tags": None,  # Allow access to all available tools
            "match_all_tags": False,  # Not used when available_tool_tags is None
            "verbose": True
        }
        if max_iterations > 0:
            agent_settings["max_iterations"] = max_iterations
            print(f"Max Iterations: {max_iterations}")
        else:
            agent_settings["max_iterations"] = None
            print("Max Iterations: Unlimited")

        ai_agent = Agent(**agent_settings)
        logger.info(f"Agent initialized successfully.")

    except Exception as e:
        logger.error(f"Failed to initialize Agent: {e}", exc_info=True)
        print(f"Error: Could not initialize Agent. Details: {e}")
        return

    print("\n--- INTERACTIVE AGENT CHAT START ---")
    print("Type 'exit', 'quit', or 'bye' to end the chat.")
    print("Type 'goals' to see current agent goals.")
    print("Type 'tools' to see available tools.")

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except UnicodeDecodeError:
                print("Error: Please use UTF-8 compatible input.")
                continue
                
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Exiting chat. Goodbye!")
                break
            
            if user_input.lower() == "goals":
                if hasattr(ai_agent, 'goal_manager') and hasattr(ai_agent.goal_manager, 'get_goals_for_display'):
                    print(ai_agent.goal_manager.get_goals_for_display())
                else:
                    print("Agent goal display not currently implemented in this version.")
                continue

            if user_input.lower() == "tools":
                tool_names = [schema['function']['name'] for schema in ai_agent.active_tools_schemas if schema.get('function') and schema['function'].get('name')]
                print("Available tools:", tool_names)
                continue

            if not user_input:
                continue

            print("Agent is thinking...")
            try:
                final_response_content = ai_agent.add_user_message_and_run(user_input)
                print(f"\nAssistant: {final_response_content}")
            except Exception as e:
                logger.error(f"Error during agent interaction: {e}", exc_info=True)
                print(f"An error occurred: {e}")

    except KeyboardInterrupt:
        print("\nInteractive chat interrupted by user. Exiting.")
    finally:
        logger.info("Interactive chat session ended.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run the LLMFlow agent")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Name of the LLM model to use"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Maximum number of agent iterations per user message. 0 for unlimited."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_interactive_chat(model_name=args.model, max_iterations=args.max_iterations) 