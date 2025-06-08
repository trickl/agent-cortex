from typing import Dict, Optional, List, Any
import requests
import os
import yaml
from abc import ABC, abstractmethod
import json

# Attempt to import openai, if not available, OPENAI_SDK_AVAILABLE will be False
try:
    from openai import OpenAI
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    # 'requests' will be imported in GenericProvider if needed

class LLMProviderInterface(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, Any]], model: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """Generate a response from the LLM based on the messages and tools.

        Args:
            messages (List[Dict[str, Any]]): A list of message objects (e.g., {"role": "user", "content": "..."}).
            model (str): The model identifier to use.
            tools (Optional[List[Dict[str, Any]]]): Optional list of tool schemas for function calling.
            **kwargs: Additional parameters (e.g., max_tokens, temperature).

        Returns:
            Dict: A dictionary containing the LLM's response, typically including
                  'role', 'content', and optionally 'tool_calls'.
                  Example: {"role": "assistant", "content": "Hello!", "tool_calls": None}
                           {"role": "assistant", "content": None, "tool_calls": [...]}
                  Returns {'role': 'assistant', 'content': 'Error: ...'} in case of error.
        """
        pass

class OpenAIProvider(LLMProviderInterface):
    """Provider for OpenAI LLMs using the official SDK."""
    def __init__(self, api_key: str):
        if not OPENAI_SDK_AVAILABLE:
            raise ImportError("OpenAI SDK not found. Please install it using 'pip install openai'.")
        if not api_key:
            raise ValueError("API_KEY must be provided for OpenAIProvider.")
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, messages: List[Dict[str, Any]], model: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 2048)
            top_p = kwargs.get("top_p", 1.0)

            request_params = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto" # Let OpenAI decide whether to use tools

            chat_completion: ChatCompletion = self.client.chat.completions.create(**request_params)
            
            if chat_completion.choices and len(chat_completion.choices) > 0:
                choice = chat_completion.choices[0]
                response_message = choice.message
                
                response_content = response_message.content
                response_tool_calls = None

                if response_message.tool_calls:
                    response_tool_calls = []
                    for tool_call in response_message.tool_calls:
                        response_tool_calls.append({
                            "id": tool_call.id,
                            "type": tool_call.type, # should be 'function'
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                
                return {
                    "role": "assistant", 
                    "content": response_content, 
                    "tool_calls": response_tool_calls
                }
            else:
                return {"role": "assistant", "content": "Error: OpenAI API call did not return choices."}
        except Exception as e:
            print(f"[OpenAIProvider] Error: {str(e)}")
            return {"role": "assistant", "content": f"Error: OpenAI API error: {str(e)}"}

class GenericProvider(LLMProviderInterface):
    """Generic provider for LLMs (e.g., Ollama, Gemini via requests)."""
    def __init__(self, config: dict):
        self.api_key = config.get('api_key') # Handled by specific logic below
        self.endpoint = config.get('endpoint')
        self.headers = config.get('headers', {}).copy() # Use a copy
        self.payload_template = config.get('payload_template', {}).copy() # Use a copy
        self.response_mapping = config.get('response_mapping', { # Default basic mapping
            "content_path": ["choices", 0, "message", "content"], # OpenAI-like default
            "error_path": ["error", "message"],
            "tool_calls_path": ["choices", 0, "message", "tool_calls"] 
        })

        if not self.endpoint:
            raise ValueError("Endpoint must be provided in config for GenericProvider.")
        
        # Specific API key handling
        # Example for Gemini: Key goes into URL param
        if 'generativelanguage.googleapis.com' in self.endpoint:
            gemini_api_key = config.get('api_key') or os.getenv("API_KEY_GENERIC") # Prioritize config
            if not gemini_api_key:
                 raise ValueError("API_KEY must be provided for Gemini in GenericProvider (config: api_key or env: API_KEY_GENERIC).")
            self.endpoint = f"{self.endpoint}?key={gemini_api_key}" # Key is part of the URL for Gemini
        elif self.api_key and "Authorization" not in self.headers : # For APIs using Bearer token
             self.headers["Authorization"] = f"Bearer {self.api_key}"


    def _get_value_from_path(self, data: Dict, path: List[str | int]) -> Any:
        """Safely get a value from a nested dict using a path list."""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
                current = current[key]
            else:
                return None
        return current

    def _deep_replace_placeholders(self, template_obj: Any, replacements: Dict[str, Any]) -> Any:
        """Recursively replaces placeholders like '{{key}}' in a nested structure."""
        if isinstance(template_obj, dict):
            return {k: self._deep_replace_placeholders(v, replacements) for k, v in template_obj.items()}
        elif isinstance(template_obj, list):
            return [self._deep_replace_placeholders(item, replacements) for item in template_obj]
        elif isinstance(template_obj, str):
            for placeholder, value in replacements.items():
                if template_obj == f"{{{{{placeholder}}}}}": # Exact match for placeholder
                    return value # Replace with the actual value (could be list, dict, etc.)
            return template_obj # No placeholder found or not an exact match
        else:
            return template_obj

    def generate_response(self, messages: List[Dict[str, Any]], model: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        payload = self._deep_replace_placeholders(self.payload_template, {"messages": messages, "prompt": messages[-1]["content"] if messages else ""})

        # Ensure model is in payload if not part of endpoint and if 'model' key exists in template
        if 'model' in payload and payload['model'] == "{{model}}": # If model is a placeholder
            payload['model'] = model
        elif 'model' not in payload and 'model' in self.payload_template: # If model key exists but was not replaced
             payload['model'] = model


        # Add/override common generation parameters from kwargs or defaults in payload_template
        # This part needs to be flexible based on how payload_template is structured
        # Example for 'generationConfig' style (like Gemini)
        if "generationConfig" in payload:
            gen_config = payload.get("generationConfig", {})
            gen_config["temperature"] = kwargs.get("temperature", gen_config.get("temperature", 0.7))
            gen_config["maxOutputTokens"] = kwargs.get("max_tokens", gen_config.get("maxOutputTokens", 2048))
            if "topP" in gen_config or "top_p" in kwargs : gen_config["topP"] = kwargs.get("top_p", gen_config.get("topP", 0.95))
            if "topK" in gen_config or "top_k" in kwargs : gen_config["topK"] = kwargs.get("top_k", gen_config.get("topK", 40))
            payload["generationConfig"] = gen_config
        else: # For OpenAI-like generic payloads not using generationConfig directly
            payload["temperature"] = kwargs.get("temperature", payload.get("temperature", 0.7))
            payload["max_tokens"] = kwargs.get("max_tokens", payload.get("max_tokens", 2048))
            if "top_p" in payload or "top_p" in kwargs: payload["top_p"] = kwargs.get("top_p", payload.get("top_p", 1.0))
        
        if tools:
            # How tools are passed to generic APIs varies wildly.
            # Some might take a 'tools' array directly, others need it in the prompt.
            # This is a placeholder; you'd need to adapt payload_template and this logic.
            if "tools" in payload and payload["tools"] == "{{tools}}":
                payload["tools"] = tools
                if "tool_choice" in payload and payload["tool_choice"] == "{{tool_choice}}":
                    payload["tool_choice"] = "auto" # Or a specific tool
            else:
                print(f"[GenericProvider] Warning: Tools provided, but payload_template doesn't have a '{{{{tools}}}}' placeholder. Tool descriptions might need to be manually added to the prompt content.")


        print(f"[GenericProvider] Sending request to: {self.endpoint}")
        print(f"[GenericProvider] Payload: {json.dumps(payload, indent=2)}")
        print(f"[GenericProvider] Headers: {self.headers}")

        try:
            # Ensure requests is imported
            if 'requests' not in globals():
                import requests as req_dynamic # Avoid conflict if already imported
            else:
                req_dynamic = requests

            response = req_dynamic.post(self.endpoint, json=payload, headers=self.headers, timeout=kwargs.get("timeout", 60))
            response.raise_for_status()
            response_data = response.json()
            print(f"[GenericProvider] Raw Response: {json.dumps(response_data, indent=2)}")

            content = self._get_value_from_path(response_data, self.response_mapping.get("content_path", []))
            tool_calls = self._get_value_from_path(response_data, self.response_mapping.get("tool_calls_path", []))
            
            # Basic type checking for safety
            if not isinstance(content, (str, type(None))): content = str(content) if content is not None else None
            if not isinstance(tool_calls, (list, type(None))): tool_calls = None


            return {"role": "assistant", "content": content, "tool_calls": tool_calls}

        except requests.RequestException as e:
            error_message = f"Generic API request error: {str(e)}"
            if e.response is not None:
                try:
                    error_details = e.response.json()
                    api_err = self._get_value_from_path(error_details, self.response_mapping.get("error_path", ["error"]))
                    if api_err: error_message += f" (Details: {api_err})"
                except json.JSONDecodeError:
                    error_message += f" (Raw response: {e.response.text})"
            print(f"[GenericProvider] Error: {error_message}")
            return {"role": "assistant", "content": f"Error: {error_message}"}
        except Exception as e:
            print(f"[GenericProvider] Unexpected error: {str(e)}")
            return {"role": "assistant", "content": f"Error: Unexpected error in GenericProvider: {str(e)}"}


class LLMClient:
    """Client for interacting with LLMs using a configured provider."""
    def __init__(self, config_file: str = "llm_config.yaml"):
        try:
            with open(config_file, 'r') as file:
                config_data = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"LLM configuration file '{config_file}' not found in the project root.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file '{config_file}': {e}")

        self.provider_config_data = config_data.get('provider_config')
        if not self.provider_config_data:
            raise ValueError(f"provider_config not found in '{config_file}'")

        provider_name = self.provider_config_data.get("provider", "generic").lower()
        # API key can be in provider_config or an environment variable
        # OpenAIProvider specifically looks for OPENAI_API_KEY env var if not in its direct api_key param
        self.api_key = self.provider_config_data.get("api_key") 
        self.model = self.provider_config_data.get("model", "default-model") # Default model if not specified

        if provider_name == "openai":
            if not OPENAI_SDK_AVAILABLE:
                print("Warning: OpenAI provider specified, but OpenAI SDK not found. Install with 'pip install openai'. Falling back to GenericProvider if endpoint is configured for it.")
                # Attempt to use GenericProvider if OpenAI SDK is missing but generic config is suitable
                if self.provider_config_data.get("endpoint"):
                    print(f"Attempting to use GenericProvider for provider 'openai' due to missing SDK. Ensure endpoint '{self.provider_config_data.get('endpoint')}' is compatible.")
                    self.provider = GenericProvider(self.provider_config_data)
                    self.model = self.provider_config_data.get("model") # GenericProvider might handle model differently
                else:
                    raise ImportError("OpenAI SDK not found and no fallback endpoint configured in provider_config for GenericProvider.")
            else:
                # For OpenAIProvider, api_key from config is passed directly.
                # If not in config, OpenAIProvider's __init__ should ideally check os.getenv("OPENAI_API_KEY")
                # For now, we ensure it's passed if available in config.
                openai_api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                     raise ValueError("API key for OpenAI must be provided in config (api_key) or OPENAI_API_KEY env var.")
                self.provider = OpenAIProvider(api_key=openai_api_key)
                print(f"Using OpenAIProvider with model: {self.model}")
        
        elif provider_name == "generic":
            print(f"Using GenericProvider for model: {self.model}. Ensure llm_config.yaml has correct endpoint, headers, payload_template, and response_mapping.")
            self.provider = GenericProvider(self.provider_config_data)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}. Supported providers are 'openai', 'generic'.")

    def generate(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """
        Generates a response from the configured LLM provider.

        Args:
            messages: A list of message objects for the LLM.
            tools: An optional list of tool schemas.
            **kwargs: Additional parameters to pass to the provider's generate_response method
                      (e.g., temperature, max_tokens). Can also override 'model'.

        Returns:
            A dictionary containing the LLM's response.
        """
        current_model = kwargs.pop("model", self.model) # Allow overriding model via kwargs
        
        # Ensure essential kwargs like temperature and max_tokens have defaults if not provided
        # and not already handled by the provider itself from its config.
        # Providers are generally expected to handle their defaults or use what's in payload_template.
        # kwargs.setdefault("temperature", 0.7)
        # kwargs.setdefault("max_tokens", 2048)

        print(f"[LLMClient] Generating response with provider: {self.provider.__class__.__name__}, Model: {current_model}")
        return self.provider.generate_response(messages, model=current_model, tools=tools, **kwargs)

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    # Create a dummy llm_config.yaml for testing if it doesn't exist
    if not os.path.exists("llm_config.yaml"):
        print("Creating dummy llm_config.yaml for testing.")
        dummy_config_content = """
provider_config:
  provider: "openai" # or "generic"
  # api_key: "YOUR_OPENAI_API_KEY" # Must be set via env var OPENAI_API_KEY if not here
  model: "gpt-4o-mini" # For OpenAI

  # Example for GenericProvider (e.g. Ollama, if it had a compatible generic endpoint)
  # provider: "generic"
  # endpoint: "http://localhost:11434/api/chat" # Ollama's chat endpoint
  # model: "llama3" # Model for Ollama
  # headers:
  #   Content-Type: "application/json"
  # payload_template:
  #   model: "{{model}}" # Placeholder for model
  #   messages: "{{messages}}" # Placeholder for messages list
  #   stream: false
  # response_mapping:
  #   content_path: ["message", "content"]
  #   error_path: ["error"]
"""
        with open("llm_config.yaml", "w") as f:
            f.write(dummy_config_content)
        print("Dummy llm_config.yaml created. Make sure to set OPENAI_API_KEY environment variable if using OpenAI provider and no key in file.")

    try:
        print(f"OpenAI SDK Available: {OPENAI_SDK_AVAILABLE}")
        # Test with OpenAI (ensure OPENAI_API_KEY is set in your environment if api_key is not in llm_config.yaml)
        # Check if config is for OpenAI, otherwise skip this test or adapt
        test_config_path = "llm_config.yaml"
        temp_conf_data = {}
        if os.path.exists(test_config_path):
             with open(test_config_path, 'r') as file:
                temp_conf_data = yaml.safe_load(file).get("provider_config", {})
        
        if temp_conf_data.get("provider") == "openai" and OPENAI_SDK_AVAILABLE:
            print("\n--- Testing OpenAIProvider ---")
            if not (temp_conf_data.get("api_key") or os.getenv("OPENAI_API_KEY")):
                print("Skipping OpenAIProvider test: API key not found in config or OPENAI_API_KEY env var.")
            else:
                try:
                    client = LLMClient(config_file=test_config_path)
                    test_messages = [{"role": "user", "content": "Hello, what is 2+2?"}]
                    # Example tool schema
                    test_tool = [{
                        "type": "function",
                        "function": {
                            "name": "calculate_sum",
                            "description": "Calculates the sum of two numbers.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "number", "description": "First number"},
                                    "b": {"type": "number", "description": "Second number"}
                                },
                                "required": ["a", "b"]
                            }
                        }
                    }]
                    response = client.generate(messages=test_messages, tools=test_tool, temperature=0.5)
                    print("\nOpenAI Response:")
                    print(json.dumps(response, indent=2))

                    # Test tool usage query
                    test_messages_tool = [{"role": "user", "content": "What is the sum of 5 and 7?"}]
                    response_tool = client.generate(messages=test_messages_tool, tools=test_tool)
                    print("\nOpenAI Response (Tool query):")
                    print(json.dumps(response_tool, indent=2))
                    if response_tool.get("tool_calls"):
                        print("Tool call requested by OpenAI.")
                    else:
                        print("No tool call requested by OpenAI for this query.")

                except Exception as e:
                    print(f"Error during OpenAIProvider test: {e}")
        else:
            print("Skipping OpenAIProvider test as provider is not 'openai' in config or SDK not available.")

        # Test with GenericProvider (using Ollama config as an example)
        # This requires Ollama to be running and 'llm_config.yaml' to be set for it.
        # For this example, we'll assume the dummy config is set to generic/Ollama
        # To actually run this, you'd change the dummy_config_content or your llm_config.yaml
        if temp_conf_data.get("provider") == "generic" and temp_conf_data.get("endpoint", "").startswith("http://localhost:11434"):
            print("\n--- Testing GenericProvider (Ollama Example) ---")
            try:
                # Make sure llm_config.yaml is configured for a generic provider (e.g. Ollama)
                # For example:
                # provider_config:
                #   provider: "generic"
                #   endpoint: "http://localhost:11434/api/chat"
                #   model: "llama3" 
                #   headers:
                #     Content-Type: "application/json"
                #   payload_template:
                #     model: "{{model}}"
                #     messages: "{{messages}}"
                #     stream: false
                #   response_mapping:
                #     content_path: ["message", "content"]
                #     error_path: ["error"]

                generic_client = LLMClient(config_file=test_config_path)
                test_messages_generic = [{"role": "user", "content": "Why is the sky blue?"}]
                response_generic = generic_client.generate(messages=test_messages_generic)
                print("\nGenericProvider Response (Ollama Example):")
                print(json.dumps(response_generic, indent=2))
            except Exception as e:
                print(f"Error during GenericProvider (Ollama) test: {e}. Ensure Ollama is running and config is correct.")
        else:
            print("Skipping GenericProvider (Ollama) test as provider is not 'generic' or endpoint is not for Ollama in current config.")
            
    except Exception as e:
        print(f"An error occurred in the example usage: {e}") 