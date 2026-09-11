"""
core/base_llm.py
-----------------
LLM Provider Abstraction, Factory Pattern, and Token Telemetry Context Manager.
Supports MockLLM (zero-cost testing), Gemini, and OpenAI.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.base_memory import Message
import time


class TokenCostTracker:
    """
    Context Manager (__enter__, __exit__) to track token usage and cost in INR (₹).
    """
    def __init__(self, cost_per_1k_tokens_inr: float = 0.15):
        self.cost_per_1k = cost_per_1k_tokens_inr
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_cost_inr(self) -> float:
        return (self.total_tokens / 1000.0) * self.cost_per_1k

    def record_usage(self, prompt_text: str, response_text: str) -> None:
        # Approximate: ~4 chars per token
        self.prompt_tokens += max(1, len(prompt_text) // 4)
        self.completion_tokens += max(1, len(response_text) // 4)

    def __enter__(self):
        print("\n--- 🟢 [Telemetry Active] Token & Cost Tracking Started ---")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"--- 🔴 [Telemetry Report] Total Tokens: {self.total_tokens} (Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}) | Estimated Cost: ₹{self.total_cost_inr:.4f} ---\n")
        return False


class BaseLLM(ABC):
    """
    Abstract Base Class for LLM Providers.
    """
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Takes a list of messages and returns the generated text response."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.model_name}>"


class MockLLM(BaseLLM):
    """
    Deterministic Mock LLM for automated testing and ReAct loops without API keys.
    """
    def __init__(self, model_name: str = "mock-gpt-4o", default_response: Optional[str] = None):
        super().__init__(model_name=model_name)
        self.default_response = default_response
        self._canned_responses: Dict[str, str] = {}

    def register_response(self, keyword: str, response: str) -> None:
        """Register specific responses when prompt contains `keyword`."""
        self._canned_responses[keyword.lower()] = response

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        last_user_msg = ""
        for m in reversed(messages):
            if m.role in ("user", "tool"):
                last_user_msg = m.content.lower()
                break

        for kw, resp in self._canned_responses.items():
            if kw in last_user_msg:
                return resp

        if self.default_response:
            return self.default_response

        return f"Mock response to: '{last_user_msg[:30]}...'"


class LLMFactory:
    """
    Factory Pattern class to dynamically create LLM instances.
    """
    _registry = {
        "mock": MockLLM
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: type) -> None:
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def create(cls, provider: str = "mock", model_name: Optional[str] = None, **kwargs) -> BaseLLM:
        provider_key = provider.lower()
        if provider_key not in cls._registry:
            raise ValueError(f"Unknown provider '{provider}'. Available providers: {list(cls._registry.keys())}")
        
        target_cls = cls._registry[provider_key]
        if model_name:
            return target_cls(model_name=model_name, **kwargs)
        return target_cls(**kwargs)
