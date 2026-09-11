"""
core/base_tool.py
------------------
The foundational Tool System for the Modular Multi-Agent AI Framework.
Defines the BaseTool abstract interface, FunctionTool wrapper, and @tool decorator.
Uses inspect.signature() for real parameter schema reflection.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Optional, List
import inspect


class BaseTool(ABC):
    """
    Abstract Base Class for all tools.
    Every tool must implement `name`, `description`, and `execute()`.
    """
    def __init__(self, name: str, description: str):
        self.name = name.strip()
        self.description = description.strip()

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool with given arguments and return a string result."""
        pass

    def to_schema(self) -> Dict[str, Any]:
        """
        Returns the metadata dictionary used by the LLM to understand how to call this tool.
        """
        return {
            "name": self.name,
            "description": self.description
        }

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allows calling the tool directly like a function: tool_instance(...)"""
        return self.execute(*args, **kwargs)


# Type name mapping for cleaner schema output
_TYPE_NAME_MAP = {
    int: "int",
    float: "float",
    str: "str",
    bool: "bool",
    list: "list",
    dict: "dict",
}


def _get_type_name(annotation: Any) -> str:
    """Convert a type annotation to a clean string name."""
    if annotation is inspect.Parameter.empty:
        return "any"
    return _TYPE_NAME_MAP.get(annotation, str(annotation))


class FunctionTool(BaseTool):
    """
    Concrete Tool implementation that wraps any standard Python function.
    Uses inspect.signature() to auto-extract parameter schemas (name, type, required).
    """
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "No description provided.").strip()
        super().__init__(name=tool_name, description=tool_desc)

        # Extract parameter schema via inspect.signature()
        self._param_schema = self._extract_param_schema(func)

    @staticmethod
    def _extract_param_schema(func: Callable) -> Dict[str, Dict[str, Any]]:
        """Uses inspect.signature() to extract real parameter names, types, and defaults."""
        sig = inspect.signature(func)
        schema: Dict[str, Dict[str, Any]] = {}
        for param_name, param in sig.parameters.items():
            param_info: Dict[str, Any] = {
                "type": _get_type_name(param.annotation),
                "required": param.default is inspect.Parameter.empty,
            }
            if param.default is not inspect.Parameter.empty:
                param_info["default"] = param.default
            schema[param_name] = param_info
        return schema

    def to_schema(self) -> Dict[str, Any]:
        """
        Returns enriched schema with parameter details for LLM tool-call formatting.
        Example output:
        {
            "name": "calculate_sip_returns",
            "description": "Calculates total estimated future value...",
            "parameters": {
                "monthly_investment": {"type": "float", "required": True},
                "annual_rate": {"type": "float", "required": True},
                "years": {"type": "int", "required": True}
            }
        }
        """
        base = super().to_schema()
        base["parameters"] = self._param_schema
        return base

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the wrapped Python function safely and returns the output."""
        try:
            result = self.func(*args, **kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to easily transform any Python function into an Agent Tool.

    Usage:
        @tool
        def add(a: int, b: int) -> int:
            '''Adds two numbers together.'''
            return a + b
    """
    def decorator(fn: Callable) -> FunctionTool:
        return FunctionTool(func=fn, name=name, description=description)

    if func is None:
        # Called with arguments: @tool(name="custom_name")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)
