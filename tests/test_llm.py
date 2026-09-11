import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.base_memory import Message
from core.base_llm import LLMFactory, TokenCostTracker

def test_llm_factory_and_telemetry():
    print("--- 1. Testing LLMFactory (Factory Pattern) ---")
    llm = LLMFactory.create("mock", model_name="mock-claude-3-5-sonnet")
    print("Created LLM instance:", llm)

    llm.register_response("tata motors", "TATA Motors stock is showing strong bullish momentum.")
    llm.register_response("sbi bank", "State Bank of India (SBI) reported a 15% increase in net profit.")

    print("\n--- 2. Testing Context Manager Telemetry (__enter__ / __exit__) ---")
    with TokenCostTracker(cost_per_1k_tokens_inr=0.20) as tracker:
        # Prompt 1
        prompt1 = "What is the status of TATA Motors?"
        messages1 = [Message(role="user", content=prompt1)]
        res1 = llm.generate(messages1)
        tracker.record_usage(prompt1, res1)
        print(f"User: {prompt1}")
        print(f"AI:   {res1}")

        # Prompt 2
        prompt2 = "Tell me about SBI Bank Q3 earnings."
        messages2 = [Message(role="user", content=prompt2)]
        res2 = llm.generate(messages2)
        tracker.record_usage(prompt2, res2)
        print(f"\nUser: {prompt2}")
        print(f"AI:   {res2}")

if __name__ == "__main__":
    test_llm_factory_and_telemetry()
    print("✅ All Milestone 3 LLM Factory tests passed successfully!")
