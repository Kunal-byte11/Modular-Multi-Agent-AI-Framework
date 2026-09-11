"""
core/base_llm.py
-----------------
Robust LLM Provider Abstraction.
Accurate endpoints, active 2026 model names, clean error diagnostics, and automatic retries.
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
    Groq Cloud API Provider.
    Uses standard browser User-Agent to bypass Cloudflare 1010 block.
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
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="ignore")
                if model != models_to_try[-1] and e.code in (404, 410, 503):
                    continue
                raise RuntimeError(f"Groq API Error ({e.code}): {err_msg}")
            except Exception as ex:
                raise RuntimeError(f"Groq Request Failed: {str(ex)}")

        raise RuntimeError("Groq request failed across all models.")


class NvidiaLLM(BaseLLM):
    """
    NVIDIA NIM API Provider.
    Uses active, non-deprecated model endpoints.
    """
    def __init__(self, model_name: str = "meta/llama-3.3-70b-instruct", api_key: Optional[str] = None, temperature: float = 0.6):
        key = api_key or os.getenv("NVIDIA_API_KEY")
        super().__init__(model_name=model_name, temperature=temperature, api_key=key)

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is missing! Set it in your .env or sidebar.")

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        payload_messages = [{"role": m.role if m.role != "tool" else "user", "content": m.content} for m in messages]
        models_to_try = [
            self.model_name,
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-8b-instruct",
            "mistralai/mistral-large-2-instruct"
        ]

        for model in models_to_try:
            payload = {
                "model": model,
                "messages": payload_messages,
                "temperature": self.temperature,
                "max_tokens": 1024
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="ignore")
                if e.code in (404, 410) and model != models_to_try[-1]:
                    continue
                raise RuntimeError(f"NVIDIA NIM API Error ({e.code}): {err_msg}")
            except Exception as ex:
                raise RuntimeError(f"NVIDIA Request Failed: {str(ex)}")

        raise RuntimeError("NVIDIA NIM request failed.")


class GeminiLLM(BaseLLM):
    """
    Google Gemini Provider.
    Queries the official Google AI Studio v1beta API with correct active model names.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", api_key: Optional[str] = None, temperature: float = 0.7):
        key = api_key or os.getenv("GEMINI_API_KEY")
        super().__init__(model_name=model_name, temperature=temperature, api_key=key)

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing! Set it in your .env or sidebar.")

        # Strip any prefix like 'models/' if passed by user
        clean_model = self.model_name.replace("models/", "")
        models_to_try = [
            clean_model,
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-pro"
        ]

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

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        last_error = None
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if "candidates" in data and len(data["candidates"]) > 0:
                        parts = data["candidates"][0]["content"]["parts"]
                        return "".join([p.get("text", "") for p in parts])
                    return "No response generated."
            except urllib.error.HTTPError as e:
                last_error = e.read().decode("utf-8", errors="ignore")
                if e.code in (404, 503):
                    continue  # Try next model candidate
                raise RuntimeError(f"Gemini API Error ({e.code}): {last_error}")
            except Exception as ex:
                last_error = str(ex)
                continue

        raise RuntimeError(f"Gemini API Error: Could not connect with tested models. Details: {last_error}")


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
