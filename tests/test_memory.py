import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.base_memory import Message, SlidingWindowMemory, SemanticMemory


def test_message_creation():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.timestamp > 0


def test_message_to_dict():
    msg = Message(role="assistant", content="Hi there")
    d = msg.to_dict()
    assert d["role"] == "assistant"
    assert d["content"] == "Hi there"


def test_sliding_window_len():
    mem = SlidingWindowMemory(max_messages=3)
    mem.add_user_message("msg1")
    mem.add_user_message("msg2")
    assert len(mem) == 2


def test_sliding_window_getitem():
    mem = SlidingWindowMemory(max_messages=3)
    mem.add_user_message("first")
    mem.add_assistant_message("second")
    assert mem[0].content == "first"
    assert mem[-1].content == "second"


def test_sliding_window_trims():
    mem = SlidingWindowMemory(max_messages=3)
    for i in range(5):
        mem.add_user_message(f"msg{i}")
    window = mem.get_context_window()
    assert len(window) == 3
    assert window[0].content == "msg2"
    assert window[-1].content == "msg4"


def test_sliding_window_iter():
    mem = SlidingWindowMemory(max_messages=10)
    mem.add_user_message("a")
    mem.add_user_message("b")
    messages = list(mem)
    assert len(messages) == 2


def test_memory_clear():
    mem = SlidingWindowMemory(max_messages=5)
    mem.add_user_message("test")
    assert len(mem) == 1
    mem.clear()
    assert len(mem) == 0


def test_semantic_search_returns_relevant():
    mem = SemanticMemory()
    mem.add_user_message("Kunal opened an account in SBI Bank Mumbai branch.")
    mem.add_user_message("Rahul invested in Tata Motors stock at 900.")
    mem.add_user_message("Pooja bought TCS shares for long term.")
    mem.add_user_message("SBI Bank interest rate is 6.5 percent.")

    results = mem.search_relevant("What bank did Kunal choose?", top_k=2)
    assert len(results) > 0
    # SBI/Kunal/bank messages should rank higher
    assert any("SBI" in r.content or "Kunal" in r.content for r in results)


def test_semantic_search_empty_query():
    mem = SemanticMemory()
    mem.add_user_message("Some content here.")
    results = mem.search_relevant("x", top_k=2)
    # Short query "x" won't match anything meaningful
    assert isinstance(results, list)


def test_memory_repr():
    mem = SlidingWindowMemory(max_messages=3)
    mem.add_user_message("test")
    assert "SlidingWindowMemory" in repr(mem)
    assert "1 messages" in repr(mem)
