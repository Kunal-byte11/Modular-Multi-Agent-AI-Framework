"""
core/base_memory.py
--------------------
Memory management subsystem for the Modular Multi-Agent AI Framework.
Implements Message schema, BaseMemory ABC, SlidingWindowMemory, and SemanticMemory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import math


@dataclass
class Message:
    """
    Structured container for agent communication messages.
    """
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    def __repr__(self) -> str:
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return f"Message({self.role.upper()}: '{preview}')"


class BaseMemory(ABC):
    """
    Abstract Base Class for Agent Memory.
    Implements standard Python Sequence protocol (__len__, __getitem__, __iter__).
    """
    def __init__(self):
        self._messages: List[Message] = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Adds a new message to the memory store."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        return msg

    def add_user_message(self, content: str) -> Message:
        return self.add_message("user", content)

    def add_assistant_message(self, content: str) -> Message:
        return self.add_message("assistant", content)

    def add_tool_message(self, tool_name: str, content: str) -> Message:
        return self.add_message("tool", content, metadata={"tool_name": tool_name})

    @abstractmethod
    def get_context_window(self) -> List[Message]:
        """Returns the active list of messages to feed into the LLM prompt."""
        pass

    def clear(self) -> None:
        """Clears all stored messages."""
        self._messages.clear()

    # Dunder / Magic Methods for Pythonic interface
    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int) -> Message:
        return self._messages[index]

    def __iter__(self):
        return iter(self._messages)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self)} messages>"


class SlidingWindowMemory(BaseMemory):
    """
    Short-Term Memory that retains only the most recent `k` messages
    to prevent LLM context limit overflow.
    """
    def __init__(self, max_messages: int = 6):
        super().__init__()
        self.max_messages = max_messages

    def get_context_window(self) -> List[Message]:
        """Returns at most `max_messages` most recent messages."""
        if len(self._messages) <= self.max_messages:
            return list(self._messages)
        return self._messages[-self.max_messages:]


class SemanticMemory(BaseMemory):
    """
    Long-Term Memory using lightweight word-overlap heuristic with cosine-like scoring
    for fast relevance-based retrieval without external heavy dependencies.
    """
    def __init__(self, max_context: int = 5):
        super().__init__()
        self.max_context = max_context

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower().strip(".,!?:;'\"()[]{}") for w in text.split() if len(w) > 2]

    def _cosine_similarity(self, query: str, document: str) -> float:
        query_words = set(self._tokenize(query))
        doc_words = set(self._tokenize(document))
        if not query_words or not doc_words:
            return 0.0
        intersection = query_words.intersection(doc_words)
        return len(intersection) / (math.sqrt(len(query_words)) * math.sqrt(len(doc_words)))

    def search_relevant(self, query: str, top_k: int = 3) -> List[Message]:
        """Searches past messages by semantic relevance to query."""
        scored = []
        for msg in self._messages:
            score = self._cosine_similarity(query, msg.content)
            scored.append((score, msg))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [msg for score, msg in scored[:top_k] if score > 0.0]

    def get_context_window(self) -> List[Message]:
        return self._messages[-self.max_context:]
