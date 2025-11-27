"""LLM client that loads providers defined in llmflow.providers."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml

from llmflow.providers import (
    GenericProvider,
    LLMProviderInterface,
    OllamaProvider,
    OpenAIProvider,
)

class LLMClient:
    """Wrapper around the configured LLM provider."""

    def __init__(self, config_file: str = "llm_config.yaml"):
        self.config_file = config_file
        provider_config = self._load_config(config_file)
        self.model = provider_config.get("model")
        if not self.model:
            raise ValueError("llm_config.yaml must specify a 'model'.")

        provider_name = provider_config.get("provider", "").lower()
        self.provider = self._build_provider(provider_name, provider_config)
        self.default_request_timeout = provider_config.get("request_timeout")
        self.provider.validate_tool_support()

    def _load_config(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as config_file:
                data = yaml.safe_load(config_file) or {}
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"LLM configuration file '{file_path}' not found."
            ) from exc
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML in '{file_path}': {exc}"
            ) from exc

        provider_config = data.get("provider_config")
        if not provider_config:
            raise ValueError("'provider_config' section missing in llm_config.yaml.")
        return provider_config

    def _build_provider(
        self, provider_name: str, provider_config: Dict[str, Any]
    ) -> LLMProviderInterface:
        if provider_name == "openai":
            api_key = provider_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI provider selected but no API key found in config or OPENAI_API_KEY."
                )
            return OpenAIProvider(api_key=api_key)

        if provider_name == "ollama":
            base_url = provider_config.get("base_url", "http://localhost:11434")
            options = provider_config.get("options", {})
            return OllamaProvider(
                model_name=self.model,
                base_url=base_url,
                default_options=options,
            )

        if provider_name == "generic":
            return GenericProvider(provider_config)

        raise ValueError(
            f"Unsupported provider '{provider_name}'. Choose 'openai', 'ollama', or 'generic'."
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        current_model = kwargs.pop("model", self.model)
        print(
            "[LLMClient] Generating response via %s (%s)",
            self.provider.__class__.__name__,
            current_model,
        )
        if "timeout" not in kwargs and self.default_request_timeout:
            kwargs["timeout"] = self.default_request_timeout
        return self.provider.generate_response(
            messages=messages,
            model=current_model,
            tools=tools,
            **kwargs,
        )