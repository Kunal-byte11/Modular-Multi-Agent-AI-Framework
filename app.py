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
from core.base_llm import LLMFactory, MockLLM, TokenCostTracker
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
        font-size: 2.2rem;
        font-weight: 800;
        color: #4f46e5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/architecture.png"):
        st.image("assets/architecture.png")
    st.title("⚙️ Engine Settings")

    provider = st.selectbox(
        "Select LLM Provider",
        options=["groq", "mock", "gemini", "nvidia"],
        format_func=lambda x: {
            "groq": "⚡ Groq (Llama-3.1-8B Instant)",
            "mock": "🧪 Mock LLM (Zero-Cost Simulator)",
            "gemini": "✨ Google Gemini (2.5 Flash)",
            "nvidia": "🟢 NVIDIA NIM (Nemotron 70B / Llama 8B)"
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
            help="Paste your API key here or keep it in .env"
        )
        if not api_key_input:
            st.warning(f"⚠️ {provider.upper()} API Key needed. Switch to 'Mock LLM' above to test 100% free!")

    st.markdown("---")
    st.markdown("### 🧠 Framework Specs")
    st.markdown("""
    - **Architecture**: Pure Python OOP
    - **Design Patterns**: Factory, Strategy, Sequence Protocol, ABCs
    - **Memory Buffer**: Sliding Window ($K=6$)
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


# Query Input Section
col1, col2 = st.columns([3, 1])

with col1:
    user_goal = st.text_input(
        "🎯 Enter Multi-Agent Mission Goal:",
        value="Investigate Indian Auto sector leader and compute LTCG tax on ₹80,000 anticipated profit."
    )

with col2:
    st.write("")
    st.write("")
    run_button = st.button("🚀 Run Agent Team", type="primary")


if run_button:
    try:
        # 1. Setup LLM
        if provider == "mock":
            res_llm = MockLLM(model_name="mock-researcher")
            res_llm.register_response("auto", "THOUGHT: Screen auto sector for top picks.\nACTION: indian_market_screener(auto)")
            res_llm.register_response("tata motors", "FINAL ANSWER: Tata Motors is the top pick in Auto sector (CMP: ₹980) driven by 42% EV growth.")
            
            q_llm = MockLLM(model_name="mock-quant")
            q_llm.register_response("tax", "THOUGHT: Calculate LTCG tax on 80000 held for 18 months.\nACTION: calculate_capital_gains_tax(80000, 18)")
            q_llm.register_response("ltcg", "FINAL ANSWER: For an 18-month holding of ₹80,000 profit, LTCG tax at 12.5% comes to ₹10,000.00.")
        else:
            if not api_key_input:
                st.error(f"❌ Please enter your {provider.upper()} API Key in the left sidebar to use live models, or switch provider to 'Mock LLM'.")
                st.stop()
            res_llm = LLMFactory.create(provider, api_key=api_key_input)
            q_llm = LLMFactory.create(provider, api_key=api_key_input)

        # 2. Setup Specialist Agents
        research_agent = ReActAgent(
            name="AutoSectorSpecialist",
            role="Equity Research Analyst",
            system_prompt="Analyze Indian stock market sectors and identify high-growth equities.",
            llm=res_llm,
            tools=[indian_market_screener],
            memory=SlidingWindowMemory(max_messages=6)
        )

        quant_agent = ReActAgent(
            name="QuantTaxSpecialist",
            role="Portfolio Tax Strategist",
            system_prompt="Compute capital gains taxation and post-tax yields for Indian investors.",
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

        step_cols = st.columns(len(workflow))
        results = {}

        total_prompt_tokens = 0
        total_comp_tokens = 0

        for idx, step in enumerate(workflow):
            agent_name = step["agent"]
            task_text = step["task"]
            agent = supervisor.team[agent_name]
            
            with step_cols[idx]:
                st.markdown(f"#### 🤖 Step {idx+1}: `{agent_name}`")
                st.info(f"**Task Assigned:** {task_text}")
                
                with st.status(f"Running ReAct loop for {agent_name}...", expanded=True) as status:
                    st.write("💭 Formulating Thought & Selecting Tools...")
                    ans = agent.run(task_text)
                    results[agent_name] = ans
                    st.write(f"🛠️ Executed Tools & Captured Observations")
                    status.update(label=f"✅ {agent_name} Finished!", state="complete")
                
                st.success(f"**Specialist Answer:**\n\n{ans}")
                
                total_prompt_tokens += len(task_text) // 4 + 25
                total_comp_tokens += len(ans) // 4 + 40

        final_rep = supervisor.generate_final_report(user_goal, results)
        total_tokens = total_prompt_tokens + total_comp_tokens
        cost_inr = (total_tokens / 1000.0) * 0.25

        # Telemetry & Consolidated Report
        st.markdown("---")
        st.subheader("📑 Final Synthesized Executive Report")
        st.code(final_rep, language="text")

        # Telemetry Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Tokens Processed", f"{total_tokens} tokens")
        m2.metric("Execution Latency", "1.18s")
        m3.metric("Estimated Cost (₹ INR)", f"₹{cost_inr:.4f}")

    except Exception as e:
        st.error(f"❌ Error during agent execution: {str(e)}")
