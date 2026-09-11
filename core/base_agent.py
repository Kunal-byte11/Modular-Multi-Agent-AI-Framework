"""
core/base_agent.py
-------------------
Base Agent Abstraction and State Engine for the Modular Multi-Agent AI Framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.base_tool import BaseTool
from core.base_memory import BaseMemory, SlidingWindowMemory
from core.base_llm import BaseLLM, LLMFactory, TokenCostTracker


class BaseAgent(ABC):
    """
    Abstract Base Class for all AI Agents.
    Encapsulates tools registry, memory management, LLM communication,
    telemetry tracking, and execution safety.
    """
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        llm: Optional[BaseLLM] = None,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        tracker: Optional[TokenCostTracker] = None,
        max_iterations: int = 5
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm or LLMFactory.create("mock")
        self.memory = memory or SlidingWindowMemory(max_messages=10)
        self.tracker = tracker
        self.max_iterations = max_iterations
        
        # Tools Registry mapping tool_name -> BaseTool object
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for t in tools:
                self.add_tool(t)

    def add_tool(self, tool: BaseTool) -> None:
        """Register a new tool with the agent."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Fetch a registered tool by its name."""
        return self._tools.get(name)

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """List of all tool schemas available for the LLM prompt."""
        return [t.to_schema() for t in self._tools.values()]

    @abstractmethod
    def run(self, user_query: str) -> str:
        """Executes the agent logic and returns the final answer."""
        pass

    def __call__(self, user_query: str) -> str:
        """Allows executing the agent directly: agent('query')."""
        return self.run(user_query)

    def __repr__(self) -> str:
        return f"<Agent: {self.name} | Role: {self.role} | Tools: {list(self._tools.keys())}>"
