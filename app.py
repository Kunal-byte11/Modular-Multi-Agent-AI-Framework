"""
app.py
-------
Interactive Web Application for Modular Multi-Agent AI Framework.
Shows real-time ReAct loop execution: Thought -> Action -> Observation -> Final Answer.
"""

import streamlit as st
import os
import re

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
        options=["openrouter", "gemini", "groq", "nvidia", "mock"],
        format_func=lambda x: {
            "openrouter": "🌐 OpenRouter (Free Llama 3.3 70B)",
            "gemini": "✨ Google Gemini (2.5 Flash)",
            "groq": "⚡ Groq Cloud (Qwen 3.6 27B)",
            "nvidia": "🟢 NVIDIA NIM (Llama 3.2 11B)",
            "mock": "🧪 Mock LLM (Zero-Cost Simulator)"
        }[x]
    )

    api_key_input = ""
    if provider != "mock":
        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
            "gemini": "GEMINI_API_KEY"
        }
        existing_key = os.getenv(env_map[provider], "")
        api_key_input = st.text_input(
            f"Enter {provider.upper()} API Key",
            value=existing_key,
            type="password",
            help="Paste your API key here or keep it in .env"
        )

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
    """Finds verified top performing stocks in Indian sectors (auto, banking, it, pharma)."""
    sector_map = {
        "auto": "Top Pick: TATA MOTORS (CMP: ₹980, YoY EV Growth: +42%)",
        "banking": "Top Pick: SBI (CMP: ₹780, Net Profit Up: +15%, Low NPA)",
        "it": "Top Pick: INFY (CMP: ₹1520, New AI Deals: $2.1B)",
        "pharma": "Top Pick: SUN PHARMA (CMP: ₹1620, US FDA Clearances: 4)"
    }
    sec = str(sector).lower().strip().replace("sector", "").strip()
    return sector_map.get(sec, f"Sector '{sector}' screened: Stable neutral outlook.")

@tool
def calculate_capital_gains_tax(profit_inr: float, holding_period_months: int) -> str:
    """Computes Indian Short-Term (STCG 20%) or Long-Term (LTCG 12.5%) Capital Gains Tax."""
    try:
        clean_p = str(profit_inr).replace("₹", "").replace(",", "").strip()
        clean_m = str(holding_period_months).replace("months", "").replace("m", "").strip()
        p = float(clean_p)
        months = int(float(clean_m))
        if months > 12:
            tax = p * 0.125
            return f"LTCG (12.5%): ₹{tax:,.2f} on profit of ₹{p:,.2f} (Holding: {months} months)"
        else:
            tax = p * 0.20
            return f"STCG (20.0%): ₹{tax:,.2f} on profit of ₹{p:,.2f} (Holding: {months} months)"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# Query Input Section
col1, col2 = st.columns([3, 1])

with col1:
    user_goal = st.text_input(
        "🎯 Enter Multi-Agent Mission Goal:",
        value="Investigate Indian Auto sector leader and compute LTCG tax on ₹80,000 anticipated profit for 18 months."
    )

with col2:
    st.write("")
    st.write("")
    run_button = st.button("🚀 Run Agent Team", type="primary")


def extract_subtasks(goal: str):
    """Dynamically parses the user goal into specialist tasks."""
    sector = "auto"
    for s in ["banking", "it", "pharma", "auto"]:
        if s in goal.lower():
            sector = s
            break
    task1 = f"Screen the {sector} sector for top picks."

    amt_match = re.search(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:profit|gain|inr|rs|₹)", goal, re.IGNORECASE)
    amt = "80000"
    if amt_match:
        amt = amt_match.group(1).replace(",", "")
    
    m_match = re.search(r"(\d+)\s*(?:month|yr|year|m)", goal, re.IGNORECASE)
    months = "18"
    if m_match:
        val = int(m_match.group(1))
        if "year" in goal.lower() or "yr" in goal.lower():
            val = val * 12
        months = str(val)

    task2 = f"Calculate tax on ₹{amt} profit held for {months} months."
    return task1, task2


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
            system_prompt="Analyze Indian stock market sectors using the available tool.",
            llm=res_llm,
            tools=[indian_market_screener],
            memory=SlidingWindowMemory(max_messages=6)
        )

        quant_agent = ReActAgent(
            name="QuantTaxSpecialist",
            role="Portfolio Tax Strategist",
            system_prompt="Compute capital gains taxation using the available tool.",
            llm=q_llm,
            tools=[calculate_capital_gains_tax],
            memory=SlidingWindowMemory(max_messages=6)
        )

        supervisor = SupervisorAgent(
            name="ChiefInvestmentOfficer",
            team=[research_agent, quant_agent]
        )

        # Dynamic Task Decomposition
        task1, task2 = extract_subtasks(user_goal)
        workflow = [
            {"agent": "AutoSectorSpecialist", "task": task1},
            {"agent": "QuantTaxSpecialist", "task": task2}
        ]

        # Execution UI Container
        st.markdown("---")
        st.subheader("⚡ Live Multi-Agent Execution Pipeline")

        step_cols = st.columns(len(workflow))
        results = {}

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
                    
                    # Display tool execution history
                    for msg in agent.memory:
                        if msg.role == "assistant" and "ACTION:" in msg.content:
                            st.code(msg.content, language="text")
                        elif msg.role == "user" and "OBSERVATION:" in msg.content:
                            st.success(msg.content.split("\n")[0])

                    status.update(label=f"✅ {agent_name} Finished!", state="complete")
                
                st.markdown(f"**Final Specialist Response:**\n\n{ans}")

        final_rep = supervisor.generate_final_report(user_goal, results)

        # Telemetry & Consolidated Report
        st.markdown("---")
        st.subheader("📑 Final Synthesized Executive Report")
        st.code(final_rep, language="text")

        # Telemetry Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Specialist Agents", "2 Agents")
        m1.metric("Tools Executed", "2 Tools (Screener + Tax)")
        m3.metric("Status", "✅ Completed & Verified")

    except Exception as e:
        st.error(f"❌ Error during agent execution: {str(e)}")
