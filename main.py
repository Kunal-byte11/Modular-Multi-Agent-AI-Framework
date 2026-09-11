"""
main.py
--------
Master Demonstration of the Modular Multi-Agent AI Framework.
Shows:
1. Tool creation with @tool
2. Sliding window memory management
3. MockLLM reasoning simulation
4. ReAct Agents (Researcher + Quant)
5. Multi-Agent Team Supervision and reporting
6. Token and Cost Telemetry Context Manager
"""

from core.base_tool import tool
from core.base_memory import SlidingWindowMemory
from core.base_llm import MockLLM, TokenCostTracker
from agents.react_agent import ReActAgent
from agents.supervisor_agent import SupervisorAgent


# -------------------------------------------------------------
# 1. Define Specialist Tools
# -------------------------------------------------------------
@tool
def indian_market_screener(sector: str) -> str:
    """Finds top performing stocks in Indian sectors (auto, banking, it)."""
    sector_map = {
        "auto": "Top Pick: TATA MOTORS (CMP: ₹980, YoY EV Growth: +42%)",
        "banking": "Top Pick: SBI (CMP: ₹780, Net Profit Up: +15%)",
        "it": "Top Pick: INFY (CMP: ₹1520, New AI Contracts: $2.1B)"
    }
    return sector_map.get(str(sector).lower().strip(), f"No active data for sector '{sector}'.")

@tool
def calculate_capital_gains_tax(profit_inr: float, holding_period_months: int) -> str:
    """Computes Indian Short-Term (STCG 20%) or Long-Term (LTCG 12.5%) Capital Gains Tax."""
    try:
        p = float(profit_inr)
        months = int(holding_period_months)
        if months > 12:
            tax = p * 0.125
            return f"LTCG (12.5%): ₹{tax:,.2f} on profit of ₹{p:,.2f}"
        else:
            tax = p * 0.20
            return f"STCG (20.0%): ₹{tax:,.2f} on profit of ₹{p:,.2f}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# -------------------------------------------------------------
# 2. Configure LLMs for Specialist Agents
# -------------------------------------------------------------
research_llm = MockLLM(model_name="mock-researcher")
research_llm.register_response(
    "auto",
    "THOUGHT: Need to look up top Indian Auto sector stocks.\n"
    "ACTION: indian_market_screener(auto)"
)
research_llm.register_response(
    "tata motors",
    "THOUGHT: Received screener data for Tata Motors.\n"
    "FINAL ANSWER: Tata Motors is the top pick in Auto sector (CMP: ₹980) driven by 42% EV growth."
)

quant_llm = MockLLM(model_name="mock-quant")
quant_llm.register_response(
    "tax",
    "THOUGHT: User wants capital gains tax calculated on ₹80,000 held for 18 months.\n"
    "ACTION: calculate_capital_gains_tax(80000, 18)"
)
quant_llm.register_response(
    "ltcg",
    "THOUGHT: Tax calculation complete.\n"
    "FINAL ANSWER: For an 18-month holding of ₹80,000 profit, LTCG tax at 12.5% comes to ₹10,000.00."
)


# -------------------------------------------------------------
# 3. Main Multi-Agent Execution Pipeline
# -------------------------------------------------------------
def run_framework_demo():
    print("==================================================")
    print("🚀 MODULAR MULTI-AGENT AI FRAMEWORK INITIALIZING")
    print("==================================================")

    # Instantiate Specialist Agents
    research_agent = ReActAgent(
        name="AutoSectorSpecialist",
        role="Equity Research Analyst",
        system_prompt="Analyze Indian stock market sectors and identify high-growth equities.",
        llm=research_llm,
        tools=[indian_market_screener],
        memory=SlidingWindowMemory(max_messages=6)
    )

    quant_agent = ReActAgent(
        name="QuantTaxSpecialist",
        role="Portfolio Tax Strategist",
        system_prompt="Compute capital gains taxation and post-tax yields for Indian investors.",
        llm=quant_llm,
        tools=[calculate_capital_gains_tax],
        memory=SlidingWindowMemory(max_messages=6)
    )

    # Instantiate Supervisor Lead
    team_supervisor = SupervisorAgent(
        name="ChiefInvestmentOfficer",
        team=[research_agent, quant_agent]
    )

    # Multi-Agent Workflow Execution with Telemetry
    user_goal = "Investigate Indian Auto sector leader and compute LTCG tax on ₹80,000 anticipated profit."

    with TokenCostTracker(cost_per_1k_tokens_inr=0.25) as telemetry:
        workflow = [
            {"agent": "AutoSectorSpecialist", "task": "Screen the auto sector for top picks."},
            {"agent": "QuantTaxSpecialist", "task": "Calculate tax on ₹80,000 profit held for 18 months."}
        ]
        
        step_results = team_supervisor.coordinate(workflow)
        final_report = team_supervisor.generate_final_report(user_goal, step_results)
        
        telemetry.record_usage(user_goal, final_report)

    print("\n" + final_report)


if __name__ == "__main__":
    run_framework_demo()
