"""
core/base_llm.py
-----------------
LLM Provider Abstraction, Factory Pattern, and Token Telemetry Context Manager.
Supports:
- MockLLM (zero-cost testing)
- GroqLLM (Ultra-fast Llama-3.3-70B)
- NvidiaLLM (NVIDIA NIM Llama-3.1-70B)
- GeminiLLM (Google Gemini 1.5/2.0 Flash)
Zero external dependencies (uses standard library urllib.request + json).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json
import urllib.request
import urllib.error
from core.base_memory import Message


def load_env_file(filepath: str = ".env") -> None:
    """
    Lightweight zero-dependency .env loader.
    Safely reads KEY=VALUE pairs into os.environ.
    """
    if not os.path.exists(filepath):
        # Look in parent directories if running from subfolder
        parent_env = os.path.join("..", filepath)
        if os.path.exists(parent_env):
            filepath = parent_env
        else:
            return

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


# Auto-load .env upon import
load_env_file()


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
    def __init__(self, model_name: str, temperature: float = 0.7, api_key: Optional[str] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

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


class GroqLLM(BaseLLM):
    """
    Groq Cloud API Provider (Zero Dependency).
    Default Model: llama-3.3-70b-versatile
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None, temperature: float = 0.6):
        key = api_key or os.getenv("GROQ_API_KEY")
        super().__init__(model_name=model_name, temperature=temperature, api_key=key)

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing! Set it in your .env file.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload_messages = [{"role": m.role if m.role != "tool" else "user", "content": m.content} for m in messages]
        payload = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": self.temperature
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"Groq API Error ({e.code}): {err_msg}")


class NvidiaLLM(BaseLLM):
    """
    NVIDIA NIM API Provider (Zero Dependency).
    Default Model: meta/llama-3.1-70b-instruct
    """
    def __init__(self, model_name: str = "meta/llama-3.1-70b-instruct", api_key: Optional[str] = None, temperature: float = 0.6):
        key = api_key or os.getenv("NVIDIA_API_KEY")
        super().__init__(model_name=model_name, temperature=temperature, api_key=key)

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is missing! Set it in your .env file.")

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload_messages = [{"role": m.role if m.role != "tool" else "user", "content": m.content} for m in messages]
        payload = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": 1024
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"NVIDIA NIM API Error ({e.code}): {err_msg}")


class GeminiLLM(BaseLLM):
    """
    Google Gemini REST API Provider (Zero Dependency).
    Default Model: gemini-1.5-flash
    """
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None, temperature: float = 0.7):
        key = api_key or os.getenv("GEMINI_API_KEY")
        super().__init__(model_name=model_name, temperature=temperature, api_key=key)

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing! Set it in your .env file.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        # Combine messages for Gemini contents format
        contents = []
        for m in messages:
            role = "user" if m.role in ("user", "tool") else "model"
            contents.append({
                "role": role,
                "parts": [{"text": m.content}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature
            }
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            raise RuntimeError(f"Gemini API Error ({e.code}): {err_msg}")


class LLMFactory:
    """
    Factory Pattern class to dynamically create LLM instances.
    """
    _registry = {
        "mock": MockLLM,
        "groq": GroqLLM,
        "nvidia": NvidiaLLM,
        "gemini": GeminiLLM
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: type) -> None:
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def create(cls, provider: str = "mock", model_name: Optional[str] = None, **kwargs) -> BaseLLM:
        provider_key = provider.lower()
        if provider_key not in cls._registry:
            raise ValueError(f"Unknown provider '{provider}'. Available: {list(cls._registry.keys())}")
        
        target_cls = cls._registry[provider_key]
        if model_name:
            return target_cls(model_name=model_name, **kwargs)
        return target_cls(**kwargs)
