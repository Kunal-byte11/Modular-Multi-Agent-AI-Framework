"""
core/base_llm.py
-----------------
LLM Provider Abstraction, Factory Pattern, and Token Telemetry Context Manager.
Includes automatic retries with exponential backoff & fallback models for 503/429 errors.
Supports:
- MockLLM (zero-cost testing)
- GroqLLM (Ultra-fast Llama-3.3-70B / Llama-3.1-8B fallback)
- NvidiaLLM (NVIDIA NIM Llama-3.1-70B / 8B fallback)
- GeminiLLM (Google Gemini 1.5/2.0 Flash with auto-retry)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json
import time
import urllib.request
import urllib.error
from core.base_memory import Message


def load_env_file(filepath: str = ".env") -> None:
    if not os.path.exists(filepath):
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


load_env_file()


class TokenCostTracker:
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
    def __init__(self, model_name: str, temperature: float = 0.7, api_key: Optional[str] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

    @abstractmethod
    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.model_name}>"


class MockLLM(BaseLLM):
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
    Groq Cloud API Provider with auto-retry and fast fallback.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None, temperature: float = 0.6):
        key = api_key or os.getenv("GROQ_API_KEY")
        super().__init__(model_name=model_name, temperature=temperature, api_key=key)

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing! Set it in your .env or sidebar.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "ModularAgentFramework/1.0"
        }
        
        payload_messages = [{"role": m.role if m.role != "tool" else "user", "content": m.content} for m in messages]
        
        models_to_try = [self.model_name, "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

        for model in models_to_try:
            payload = {
                "model": model,
                "messages": payload_messages,
                "temperature": self.temperature
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        return data["choices"][0]["message"]["content"]
                except urllib.error.HTTPError as e:
                    if e.code in (503, 429, 500) and attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    if model != models_to_try[-1]:
                        break  # Try next fallback model
                    err_msg = e.read().decode("utf-8")
                    raise RuntimeError(f"Groq API Error ({e.code}): {err_msg}")
                except Exception as ex:
                    if attempt < 2:
                        time.sleep(1.0)
                        continue
                    raise RuntimeError(f"Groq Request Failed: {str(ex)}")

        raise RuntimeError("Groq service temporarily unavailable after fallback retries.")


class NvidiaLLM(BaseLLM):
    """
    NVIDIA NIM API Provider with auto-retry.
    """
    def __init__(self, model_name: str = "meta/llama-3.1-70b-instruct", api_key: Optional[str] = None, temperature: float = 0.6):
        key = api_key or os.getenv("NVIDIA_API_KEY")
        super().__init__(model_name=model_name, temperature=temperature, api_key=key)

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is missing! Set it in your .env or sidebar.")

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "ModularAgentFramework/1.0"
        }
        
        payload_messages = [{"role": m.role if m.role != "tool" else "user", "content": m.content} for m in messages]
        models_to_try = [self.model_name, "meta/llama-3.1-8b-instruct", "mistralai/mixtral-8x7b-instruct-v0.1"]

        for model in models_to_try:
            payload = {
                "model": model,
                "messages": payload_messages,
                "temperature": self.temperature,
                "max_tokens": 1024
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        return data["choices"][0]["message"]["content"]
                except urllib.error.HTTPError as e:
                    if e.code in (503, 429, 500) and attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    if model != models_to_try[-1]:
                        break
                    err_msg = e.read().decode("utf-8")
                    raise RuntimeError(f"NVIDIA NIM API Error ({e.code}): {err_msg}")
                except Exception as ex:
                    if attempt < 2:
                        time.sleep(1.0)
                        continue
                    raise RuntimeError(f"NVIDIA Request Failed: {str(ex)}")

        raise RuntimeError("NVIDIA NIM service temporarily unavailable.")


class GeminiLLM(BaseLLM):
    """
    Google Gemini Provider with auto-retry and multi-model fallback.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None, temperature: float = 0.7):
        key = api_key or os.getenv("GEMINI_API_KEY")
        super().__init__(model_name=model_name, temperature=temperature, api_key=key)

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing! Set it in your .env or sidebar.")

        models_to_try = [self.model_name, "gemini-1.5-flash-latest", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
        headers = {"Content-Type": "application/json", "User-Agent": "ModularAgentFramework/1.0"}

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

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        if "candidates" in data and len(data["candidates"]) > 0:
                            parts = data["candidates"][0]["content"]["parts"]
                            return "".join([p.get("text", "") for p in parts])
                        return "No response generated."
                except urllib.error.HTTPError as e:
                    if e.code in (503, 429, 500) and attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    if model != models_to_try[-1]:
                        break  # Fallback to next Gemini model
                    err_msg = e.read().decode("utf-8")
                    raise RuntimeError(f"Gemini API Error ({e.code}): Model overloaded or temporary issue. {err_msg}")
                except Exception as ex:
                    if attempt < 2:
                        time.sleep(1.0)
                        continue
                    raise RuntimeError(f"Gemini Request Failed: {str(ex)}")

        raise RuntimeError("Google Gemini servers temporarily busy (503). Try switching to Groq or Mock LLM.")


class LLMFactory:
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
