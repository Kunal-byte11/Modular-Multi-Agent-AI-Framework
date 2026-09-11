"""
interactive_cli.py
-------------------
Simple, Interactive Python Terminal UI to test every component of the framework.
Zero dependencies — runs out of the box with pure Python!
"""

import sys
import os
import time

from core.base_tool import tool
from core.base_memory import SlidingWindowMemory, SemanticMemory
from core.base_llm import LLMFactory, MockLLM, TokenCostTracker
from agents.react_agent import ReActAgent
from agents.supervisor_agent import SupervisorAgent


def clear_banner():
    print("\n" + "="*60)
    print("🤖  MODULAR MULTI-AGENT AI FRAMEWORK — INTERACTIVE TEST BENCH")
    print("="*60)


# Define Common Specialist Tools
@tool
def stock_screener(sector: str) -> str:
    """Finds top performing stocks in Indian sectors (auto, banking, it)."""
    data = {
        "auto": "TATA MOTORS (CMP: ₹980 | 1Y Return: +42% | EV Leader)",
        "banking": "SBI BANK (CMP: ₹780 | 1Y Return: +28% | Low NPA)",
        "it": "INFOSYS (CMP: ₹1520 | 1Y Return: +18% | AI Deals: $2.1B)"
    }
    return data.get(str(sector).lower().strip(), f"No active data for sector '{sector}'. Try 'auto', 'banking', or 'it'.")

@tool
def calculate_tax(profit_inr: float, holding_months: int) -> str:
    """Calculates STCG (20%) or LTCG (12.5%) Indian Capital Gains Tax."""
    try:
        p = float(profit_inr)
        m = int(holding_months)
        if m > 12:
            tax = p * 0.125
            return f"LTCG Tax (12.5%): ₹{tax:,.2f} on profit of ₹{p:,.2f}"
        else:
            tax = p * 0.20
            return f"STCG Tax (20.0%): ₹{tax:,.2f} on profit of ₹{p:,.2f}"
    except Exception as e:
        return f"Tax Calculation Error: {str(e)}"

@tool
def calculate_sip(monthly_amount: float, annual_rate: float, years: int) -> str:
    """Calculates SIP future value for mutual fund investments."""
    try:
        p = float(monthly_amount)
        r = (float(annual_rate) / 100) / 12
        n = int(years) * 12
        fv = p * (((1 + r)**n - 1) / r) * (1 + r)
        invested = p * n
        gain = fv - invested
        return f"Invested: ₹{invested:,.2f} | Estimated Gain: ₹{gain:,.2f} | Total Value: ₹{fv:,.2f}"
    except Exception as e:
        return f"SIP Error: {str(e)}"


def get_mock_engine():
    res_llm = MockLLM(model_name="mock-researcher")
    res_llm.register_response("auto", "THOUGHT: Screener for auto sector.\nACTION: stock_screener(auto)")
    res_llm.register_response("banking", "THOUGHT: Screener for banking sector.\nACTION: stock_screener(banking)")
    res_llm.register_response("it", "THOUGHT: Screener for IT sector.\nACTION: stock_screener(it)")
    res_llm.register_response("tata", "FINAL ANSWER: Tata Motors is the top pick in Auto (CMP: ₹980) driven by EV growth.")
    res_llm.register_response("sbi", "FINAL ANSWER: SBI Bank is the top pick in Banking (CMP: ₹780) with rising net profit.")
    res_llm.register_response("infy", "FINAL ANSWER: Infosys is the top pick in IT (CMP: ₹1520) with $2.1B in AI contracts.")

    quant_llm = MockLLM(model_name="mock-quant")
    quant_llm.register_response("tax", "THOUGHT: Calculate capital gains tax.\nACTION: calculate_tax(80000, 18)")
    quant_llm.register_response("ltcg", "FINAL ANSWER: LTCG Tax on ₹80,000 profit for 18 months at 12.5% is ₹10,000.00.")
    quant_llm.register_response("sip", "THOUGHT: Calculate SIP returns for 5000/mo at 12% for 5 years.\nACTION: calculate_sip(5000, 12, 5)")
    quant_llm.register_response("gain", "FINAL ANSWER: A ₹5,000/mo SIP at 12% for 5 years grows to ₹4.12 Lakhs (Total Gain: ₹1.12 Lakhs).")

    researcher = ReActAgent(
        name="MarketResearcher",
        role="Equity Research Specialist",
        system_prompt="Analyze Indian market sectors.",
        llm=res_llm,
        tools=[stock_screener],
        memory=SlidingWindowMemory(max_messages=6)
    )

    quant = ReActAgent(
        name="QuantAnalyst",
        role="Quantitative Finance & Tax Strategist",
        system_prompt="Perform financial computations and tax optimization.",
        llm=quant_llm,
        tools=[calculate_tax, calculate_sip],
        memory=SlidingWindowMemory(max_messages=6)
    )

    supervisor = SupervisorAgent(
        name="InvestmentSupervisor",
        team=[researcher, quant]
    )

    return researcher, quant, supervisor


def run_interactive_menu():
    researcher, quant, supervisor = get_mock_engine()

    while True:
        clear_banner()
        print("Choose a test scenario to run:")
        print("  [1] 🔍 Test Agent 1: Market Research Specialist (ReAct + Tool Screener)")
        print("  [2] 📊 Test Agent 2: Quant & Tax Specialist (ReAct + SIP & Tax Tools)")
        print("  [3] 👔 Test Full Multi-Agent Team (Supervisor Coordination + Telemetry)")
        print("  [4] 🧠 Test Short-Term & Long-Term Memory (Sliding Window & TF-IDF)")
        print("  [5] 🔑 Test Live LLM APIs (Groq / NVIDIA / Gemini from .env)")
        print("  [6] 🌐 Launch Streamlit Web Browser UI (streamlit run app.py)")
        print("  [0] ❌ Exit")
        print("="*60)

        choice = input("Enter option [0-6]: ").strip()

        if choice == "1":
            print("\n--- Running Market Research Agent ---")
            sector = input("Enter sector to screen [auto / banking / it] (default: auto): ").strip() or "auto"
            query = f"Find the top performing stock in the {sector} sector."
            print(f"User Query: '{query}'")
            ans = researcher(query)
            print("\n🎯 Agent Output:\n" + ans)
            input("\nPress Enter to return to menu...")

        elif choice == "2":
            print("\n--- Running Quant & Tax Agent ---")
            print("  1. Calculate LTCG/STCG Tax")
            print("  2. Calculate 5-Year SIP Growth")
            sub = input("Select sub-test [1 or 2]: ").strip()
            if sub == "2":
                ans = quant("Calculate SIP returns for ₹5,000 per month at 12% for 5 years.")
            else:
                ans = quant("Calculate capital gains tax on ₹80,000 profit held for 18 months.")
            print("\n🎯 Agent Output:\n" + ans)
            input("\nPress Enter to return to menu...")

        elif choice == "3":
            print("\n--- Running Full Multi-Agent Team with Live Telemetry ---")
            goal = "Analyze Indian Auto sector leader and compute LTCG tax on ₹80,000 anticipated profit."
            with TokenCostTracker(cost_per_1k_tokens_inr=0.25) as tracker:
                workflow = [
                    {"agent": "MarketResearcher", "task": "Screen the auto sector for top picks."},
                    {"agent": "QuantAnalyst", "task": "Calculate tax on ₹80,000 profit held for 18 months."}
                ]
                results = supervisor.coordinate(workflow)
                report = supervisor.generate_final_report(goal, results)
                tracker.record_usage(goal, report)

            print("\n" + report)
            input("\nPress Enter to return to menu...")

        elif choice == "4":
            print("\n--- Testing Memory Engine ---")
            mem = SemanticMemory()
            mem.add_user_message("Kunal opened an SBI Bank savings account in Mumbai.")
            mem.add_user_message("Rahul bought Tata Motors shares at ₹900.")
            mem.add_user_message("SBI Bank interest rate is 6.5%.")
            q = "What bank does Kunal use?"
            print(f"Memory Search Query: '{q}'")
            res = mem.search_relevant(q, top_k=2)
            print(f"Top Matches ({len(res)}):")
            for m in res:
                print(" ->", m)
            input("\nPress Enter to return to menu...")

        elif choice == "5":
            from tests.test_live_llm import test_available_live_llms
            test_available_live_llms()
            input("\nPress Enter to return to menu...")

        elif choice == "6":
            print("\n🚀 Launching Streamlit Web App...")
            print("Run this command in your terminal: streamlit run app.py")
            input("Press Enter to return to menu...")

        elif choice == "0":
            print("\n👋 Exiting Interactive Test Bench. Great job!")
            break
        else:
            print("Invalid selection! Please enter a number between 0 and 6.")
            time.sleep(1)


if __name__ == "__main__":
    run_interactive_menu()
