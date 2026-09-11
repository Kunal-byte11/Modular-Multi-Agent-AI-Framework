import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.base_memory import Message
from core.base_llm import LLMFactory, TokenCostTracker

def test_available_live_llms():
    print("==================================================")
    print("🔑 TESTING LIVE LLM PROVIDERS (.env)")
    print("==================================================")

    providers = [
        ("groq", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
        ("nvidia", "NVIDIA_API_KEY", "meta/llama-3.1-70b-instruct"),
        ("gemini", "GEMINI_API_KEY", "gemini-1.5-flash")
    ]

    active_found = False

    for prov_name, env_key, model in providers:
        key_val = os.getenv(env_key)
        if key_val and not key_val.startswith("your_"):
            print(f"\n🟢 Found active key for [{prov_name.upper()}]: {key_val[:8]}...")
            active_found = True
            try:
                llm = LLMFactory.create(prov_name, model_name=model)
                with TokenCostTracker() as telemetry:
                    prompt = "Hello! In 1 short sentence, what is an AI Agent?"
                    print(f"User: {prompt}")
                    res = llm.generate([Message(role="user", content=prompt)])
                    telemetry.record_usage(prompt, res)
                    print(f"AI ({prov_name}): {res.strip()}")
            except Exception as e:
                print(f"❌ Error testing {prov_name}: {str(e)}")
        else:
            print(f"⚪ [{prov_name.upper()}]: No key detected in .env ({env_key})")

    if not active_found:
        print("\n💡 TIP: To test live models, copy .env.example to .env and paste your Groq, NVIDIA, or Gemini API key!")

if __name__ == "__main__":
    test_available_live_llms()
