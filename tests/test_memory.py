import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.base_memory import SlidingWindowMemory, SemanticMemory

def test_sliding_window_memory():
    print("--- 1. Testing Sliding Window Short-Term Memory ---")
    mem = SlidingWindowMemory(max_messages=3)
    
    mem.add_user_message("Hello, I am Kunal.")
    mem.add_assistant_message("Hi Kunal! How can I assist you?")
    mem.add_user_message("What is my name?")
    mem.add_assistant_message("Your name is Kunal.")
    mem.add_user_message("Tell me about SBI stock.")

    print(f"Total messages recorded in history: {len(mem)}") # calls __len__
    print(f"First message in history: {mem[0]}")              # calls __getitem__
    print(f"Latest message in history: {mem[-1]}")

    window = mem.get_context_window()
    print(f"\nActive Context Window (Max 3):")
    for msg in window:
        print(" ->", msg)
    assert len(window) == 3

def test_semantic_long_term_memory():
    print("\n--- 2. Testing Semantic Long-Term Retrieval ---")
    mem = SemanticMemory()
    mem.add_user_message("Kunal opened an account in SBI Bank Mumbai branch.")
    mem.add_user_message("Rahul invested in Tata Motors stock at ₹900.")
    mem.add_user_message("Pooja bought TCS shares for long term.")
    mem.add_user_message("SBI Bank interest rate is 6.5 percent.")

    results = mem.search_relevant("What bank did Kunal choose?", top_k=2)
    print("Query: 'What bank did Kunal choose?'")
    print(f"Top Retrieved Messages ({len(results)} matches):")
    for res in results:
        print(" ->", res)

if __name__ == "__main__":
    test_sliding_window_memory()
    test_semantic_long_term_memory()
    print("\n✅ All Milestone 2 Memory tests passed successfully!")
