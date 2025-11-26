"""Base interfaces for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProviderInterface(ABC):
    """Abstract base class for every LLM provider."""

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a chat completion request and return the assistant payload."""

    @abstractmethod
    def validate_tool_support(self) -> None:
        """Raise an exception if the provider/model cannot emit tool calls."""
