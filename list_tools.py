from llmflow.tools.tool_registry import list_available_tools, get_all_tools_schemas

def main():
    print("=== Available Tools ===")
    tools = list_available_tools(verbose=True)
    print("\n=== Tool Schemas ===")
    schemas = get_all_tools_schemas()
    for tool in schemas:
        print(f"\nTool: {tool['function']['name']}")
        print(f"Description: {tool['function']['description']}")
        print(f"Parameters: {tool['function']['parameters']}")

if __name__ == "__main__":
    main()
