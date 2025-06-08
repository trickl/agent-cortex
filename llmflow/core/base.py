"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Base Module - Core foundation classes for the LLMFlow framework.
This module provides the fundamental building blocks used throughout the system:
- ToolResponse: A standardized response format for all tool executions
- BaseTool: Abstract base class that all tools must inherit from

The base module ensures consistent tool behavior and error handling across the framework.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ToolResponse:
    """Response from a tool execution."""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

class BaseTool:
    """Base class for all tools."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        
    def execute(self, **kwargs) -> ToolResponse:
        """Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResponse: The result of the tool execution
        """
        try:
            result = self._execute(**kwargs)
            return ToolResponse(
                success=True,
                message="Tool executed successfully",
                data=result
            )
        except Exception as e:
            return ToolResponse(
                success=False,
                message=f"Tool execution failed: {str(e)}",
                error=str(e)
            )
    
    def _execute(self, **kwargs) -> Any:
        """Internal execution method to be implemented by subclasses.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Any: The result of the tool execution
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement _execute method") 