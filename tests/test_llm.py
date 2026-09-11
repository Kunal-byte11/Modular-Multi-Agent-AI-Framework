import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.base_memory import Message
from core.base_llm import MockLLM, LLMFactory, TokenCostTracker, BaseLLM


def test_factory_creates_mock():
    llm = LLMFactory.create("mock")
    assert isinstance(llm, MockLLM)


def test_factory_with_model_name():
    llm = LLMFactory.create("mock", model_name="custom-model")
    assert llm.model_name == "custom-model"


def test_factory_unknown_provider():
    try:
        LLMFactory.create("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown provider" in str(e)


def test_mock_llm_register_response():
    llm = MockLLM()
    llm.register_response("tata", "TATA Motors is bullish.")
    messages = [Message(role="user", content="Tell me about TATA Motors")]
    result = llm.generate(messages)
    assert "TATA Motors is bullish" in result


def test_mock_llm_default_response():
    llm = MockLLM(default_response="I am a mock.")
    messages = [Message(role="user", content="random query xyz")]
    result = llm.generate(messages)
    assert result == "I am a mock."


def test_mock_llm_fallback():
    llm = MockLLM()
    messages = [Message(role="user", content="completely random query")]
    result = llm.generate(messages)
    assert "Mock response" in result


def test_tracker_record_usage():
    tracker = TokenCostTracker(cost_per_1k_tokens_inr=0.20)
    tracker.record_usage("Hello world prompt", "Response text here")
    assert tracker.prompt_tokens > 0
    assert tracker.completion_tokens > 0
    assert tracker.total_tokens == tracker.prompt_tokens + tracker.completion_tokens


def test_tracker_cost_calculation():
    tracker = TokenCostTracker(cost_per_1k_tokens_inr=1.0)
    tracker.prompt_tokens = 500
    tracker.completion_tokens = 500
    assert tracker.total_cost_inr == 1.0


def test_tracker_context_manager():
    with TokenCostTracker() as tracker:
        tracker.record_usage("prompt", "response")
        assert tracker.total_tokens > 0
    # After exiting, tracker should still be accessible
    assert tracker.total_tokens > 0


def test_llm_repr():
    llm = MockLLM(model_name="test-model")
    assert "MockLLM" in repr(llm)
    assert "test-model" in repr(llm)
