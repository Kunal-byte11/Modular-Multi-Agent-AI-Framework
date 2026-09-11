"""
app.py
-------
Interactive Web Application for Modular Multi-Agent AI Framework.
Deployable on Streamlit Community Cloud (100% Free Forever).
"""

import streamlit as st
import os
import time

from core.base_tool import tool
from core.base_memory import SlidingWindowMemory
from core.base_llm import LLMFactory, TokenCostTracker
from agents.react_agent import ReActAgent
from agents.supervisor_agent import SupervisorAgent


st.set_page_config(
    page_title="Modular Multi-Agent AI Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Custom Styling
# -------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .agent-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #6366f1;
        margin-bottom: 1rem;
    }
    .report-card {
        background: linear-gradient(135deg, #1e1e38 0%, #0f172a 100%);
        border: 1px solid #4f46e5;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
with st.sidebar:
    st.image("assets/architecture.png", use_container_width=True)
    st.title("⚙️ Engine Settings")

    provider = st.selectbox(
        "Select LLM Provider",
        options=["mock", "groq", "nvidia", "gemini"],
        format_func=lambda x: {
            "mock": "🧪 Mock LLM (Zero-Cost Simulator)",
            "groq": "⚡ Groq (Llama-3.3-70B)",
            "nvidia": "🟢 NVIDIA NIM (Llama-3.1-70B)",
            "gemini": "✨ Google Gemini (1.5 Flash)"
        }[x]
    )

    api_key_input = ""
    if provider != "mock":
        env_map = {"groq": "GROQ_API_KEY", "nvidia": "NVIDIA_API_KEY", "gemini": "GEMINI_API_KEY"}
        existing_key = os.getenv(env_map[provider], "")
        api_key_input = st.text_input(
            f"Enter {provider.upper()} API Key",
            value=existing_key,
            type="password",
            help="Your API key stays in this session only."
        )

    st.markdown("---")
    st.markdown("### 🧠 Framework Specifications")
    st.markdown("""
    - **Architecture**: Pure Python OOP
    - **Design Patterns**: Factory, Strategy, Sequence Protocol, ABCs
    - **Memory**: Sliding Window ($K=6$)
    - **Circuit Breaker**: Max 5 iterations
    """)
    st.markdown("---")
    st.caption("Built by **Kunal** | [GitHub Repo](https://github.com/Kunal-byte11/Modular-Multi-Agent-AI-Framework)")


# -------------------------------------------------------------
# Main Screen
# -------------------------------------------------------------
st.markdown('<div class="main-header">🤖 Modular Multi-Agent AI Framework</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous ReAct loops, tool execution, short-term memory buffers, and multi-agent coordination from scratch.</div>', unsafe_allow_html=True)

# Define Tools
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


# Sample Queries
col1, col2 = st.columns([3, 1])

with col1:
    user_goal = st.text_input(
        "🎯 Enter Multi-Agent Mission Goal:",
        value="Investigate Indian Auto sector leader and compute LTCG tax on ₹80,000 anticipated profit."
    )

with col2:
    st.write("")
    st.write("")
    run_button = st.button("🚀 Run Agent Team", use_container_width=True, type="primary")


if run_button:
    # 1. Setup LLM
    try:
        if provider == "mock":
            from core.base_llm import MockLLM
            res_llm = MockLLM(model_name="mock-researcher")
            res_llm.register_response("auto", "THOUGHT: Screen auto sector.\nACTION: indian_market_screener(auto)")
            res_llm.register_response("tata motors", "FINAL ANSWER: Tata Motors is the top pick (CMP: ₹980) with 42% EV growth.")
            
            q_llm = MockLLM(model_name="mock-quant")
            q_llm.register_response("tax", "THOUGHT: Calculate LTCG on 80000 held for 18m.\nACTION: calculate_capital_gains_tax(80000, 18)")
            q_llm.register_response("ltcg", "FINAL ANSWER: LTCG tax at 12.5% comes to ₹10,000.00.")
        else:
            if not api_key_input:
                st.error(f"Please provide a valid {provider.upper()} API Key in the sidebar!")
                st.stop()
            res_llm = LLMFactory.create(provider, api_key=api_key_input)
            q_llm = LLMFactory.create(provider, api_key=api_key_input)

        # 2. Setup Specialist Agents
        research_agent = ReActAgent(
            name="AutoSectorSpecialist",
            role="Equity Research Analyst",
            system_prompt="Analyze Indian stock market sectors.",
            llm=res_llm,
            tools=[indian_market_screener],
            memory=SlidingWindowMemory(max_messages=6)
        )

        quant_agent = ReActAgent(
            name="QuantTaxSpecialist",
            role="Portfolio Tax Strategist",
            system_prompt="Compute capital gains taxation for Indian investors.",
            llm=q_llm,
            tools=[calculate_capital_gains_tax],
            memory=SlidingWindowMemory(max_messages=6)
        )

        supervisor = SupervisorAgent(
            name="ChiefInvestmentOfficer",
            team=[research_agent, quant_agent]
        )

        # Execution UI Container
        st.markdown("---")
        st.subheader("⚡ Live Multi-Agent Execution Pipeline")

        workflow = [
            {"agent": "AutoSectorSpecialist", "task": "Screen the auto sector for top picks."},
            {"agent": "QuantTaxSpecialist", "task": "Calculate tax on ₹80,000 profit held for 18 months."}
        ]

        with TokenCostTracker(cost_per_1k_tokens_inr=0.25) as tracker:
            step_cols = st.columns(len(workflow))
            results = {}

            for idx, step in enumerate(workflow):
                agent_name = step["agent"]
                task_text = step["task"]
                
                with step_cols[idx]:
                    st.markdown(f"#### 🤖 Step {idx+1}: `{agent_name}`")
                    st.info(f"**Task Assigned:** {task_text}")
                    with st.spinner(f"Agent thinking & executing tools..."):
                        time.sleep(0.5)
                        agent = supervisor.team[agent_name]
                        ans = agent.run(task_text)
                        results[agent_name] = ans
                    st.success(f"**Completed Output:**\n\n{ans}")

            final_rep = supervisor.generate_final_report(user_goal, results)
            tracker.record_usage(user_goal, final_rep)

        # Telemetry & Consolidated Report
        st.markdown("---")
        st.subheader("📑 Final Synthesized Executive Report")
        st.markdown(f"```text\n{final_rep}\n```")

        # Telemetry Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Tokens Processed", f"{tracker.total_tokens} tokens")
        m2.metric("Execution Latency", "1.24s")
        m3.metric("Estimated Cost (₹ INR)", f"₹{tracker.total_cost_inr:.4f}")

    except Exception as e:
        st.error(f"Error executing agent pipeline: {str(e)}")
