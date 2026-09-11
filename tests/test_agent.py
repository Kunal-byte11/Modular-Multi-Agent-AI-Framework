import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.base_tool import tool
from core.base_llm import MockLLM, TokenCostTracker
from agents.react_agent import ReActAgent


# --- Test tools ---

@tool
def stock_price_lookup(ticker: str) -> str:
    """Fetches real-time price of Indian stocks."""
    prices = {"TATA": "₹980", "SBI": "₹780", "RELIANCE": "₹2950"}
    return prices.get(ticker.upper(), f"₹0 (Ticker {ticker} not found)")


@tool
def tax_calculator(amount: str) -> str:
    """Calculates 10% Long Term Capital Gains tax on profit amount."""
    val = float(amount.replace("₹", "").replace(",", "").strip())
    tax = val * 0.10
    return f"₹{tax:.2f}"


def _make_agent(mock_llm=None):
    """Helper to create a test agent with MockLLM."""
    llm = mock_llm or MockLLM()
    return ReActAgent(
        name="TestAgent",
        role="Test Assistant",
        system_prompt="Help with testing.",
        llm=llm,
        tools=[stock_price_lookup, tax_calculator],
        max_iterations=4
    )


# --- Tests ---

def test_react_agent_calls_tool_and_returns_answer():
    llm = MockLLM()
    llm.register_response(
        "sbi",
        "THOUGHT: Need to look up SBI stock price.\nACTION: stock_price_lookup(SBI)"
    )
    llm.register_response(
        "780",
        "THOUGHT: Got the price.\nFINAL ANSWER: SBI stock is at ₹780 per share."
    )
    agent = _make_agent(llm)
    answer = agent("What is the current stock price of SBI?")
    assert "780" in answer
    assert "SBI" in answer


def test_react_agent_handles_missing_tool():
    llm = MockLLM(default_response="THOUGHT: Calling tool.\nACTION: nonexistent_tool(arg)")
    agent = _make_agent(llm)
    # Should not crash, should return max iterations message
    answer = agent("test query")
    assert isinstance(answer, str)


def test_react_agent_max_iterations():
    llm = MockLLM(default_response="THOUGHT: I'm thinking...")
    agent = ReActAgent(
        name="StuckAgent",
        role="Stuck",
        system_prompt="Test",
        llm=llm,
        tools=[],
        max_iterations=2
    )
    answer = agent("test")
    # With no tools, first non-action response is returned directly
    assert isinstance(answer, str)


def test_react_agent_with_tracker():
    llm = MockLLM()
    llm.register_response(
        "sbi",
        "THOUGHT: Looking up.\nACTION: stock_price_lookup(SBI)"
    )
    llm.register_response(
        "780",
        "THOUGHT: Done.\nFINAL ANSWER: SBI is ₹780."
    )
    tracker = TokenCostTracker()
    agent = ReActAgent(
        name="TrackedAgent",
        role="Test",
        system_prompt="Test",
        llm=llm,
        tools=[stock_price_lookup],
        tracker=tracker,
        max_iterations=4
    )
    agent("SBI price?")
    assert tracker.total_tokens > 0


def test_react_agent_tool_schemas_in_prompt():
    agent = _make_agent()
    schemas = agent.tool_schemas
    assert len(schemas) == 2
    assert any(s["name"] == "stock_price_lookup" for s in schemas)
    assert "parameters" in schemas[0]


def test_react_agent_json_tool_calling():
    # Test JSON-formatted action: ACTION: stock_price_lookup({"ticker": "TATA"})
    llm = MockLLM()
    llm.register_response(
        "tata",
        'THOUGHT: Looking up TATA with JSON.\nACTION: stock_price_lookup({"ticker": "TATA"})'
    )
    llm.register_response(
        "980",
        "THOUGHT: Got ₹980.\nFINAL ANSWER: Tata Motors is ₹980."
    )
    agent = _make_agent(llm)
    answer = agent("Check TATA")
    assert "980" in answer
    assert "Tata Motors" in answer


def test_supervisor_agent_decomposition_and_coordination():
    from agents.supervisor_agent import SupervisorAgent
    import json

    res_llm = MockLLM()
    res_llm.register_response("tata", "FINAL ANSWER: TATA Motors CMP is ₹980.")
    researcher = ReActAgent(name="Researcher", role="Equity Analyst", system_prompt="", llm=res_llm)

    sup_llm = MockLLM()
    canned_tasks = json.dumps([{"agent": "Researcher", "task": "Check TATA"}])
    sup_llm.register_response("decompose", canned_tasks)
    sup_llm.default_response = canned_tasks

    supervisor = SupervisorAgent(name="Lead", team=[researcher], llm=sup_llm)
    tasks = supervisor.decompose("Analyze TATA Motors")
    assert len(tasks) == 1
    assert tasks[0]["agent"] == "Researcher"

    report = supervisor.run("Analyze TATA Motors")
    assert "EXECUTIVE MULTI-AGENT REPORT" in report
    assert "TATA Motors CMP is ₹980" in report


def test_react_agent_repr():
    agent = _make_agent()
    r = repr(agent)
    assert "TestAgent" in r
    assert "stock_price_lookup" in r

