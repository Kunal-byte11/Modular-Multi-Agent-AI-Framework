import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.base_tool import BaseTool, FunctionTool, tool


# --- Fixtures: sample tools ---

@tool
def calculate_sip_returns(monthly_investment: float, annual_rate: float, years: int) -> float:
    """Calculates total estimated future value of an SIP in mutual funds."""
    months = years * 12
    monthly_rate = (annual_rate / 100) / 12
    future_value = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    return round(future_value, 2)


class StockPriceTool(BaseTool):
    def __init__(self):
        super().__init__(name="stock_price_fetcher", description="Fetches live stock price for Indian NSE/BSE tickers.")

    def execute(self, ticker: str) -> str:
        mock_prices = {"TATA": "₹980", "RELIANCE": "₹2950", "INFY": "₹1520"}
        return mock_prices.get(ticker.upper(), f"Ticker '{ticker}' not found on NSE.")


# --- Tests ---

def test_tool_decorator_creates_function_tool():
    assert isinstance(calculate_sip_returns, FunctionTool)


def test_tool_name_from_function():
    assert calculate_sip_returns.name == "calculate_sip_returns"


def test_tool_description_from_docstring():
    assert "SIP" in calculate_sip_returns.description


def test_tool_execute_returns_string():
    result = calculate_sip_returns.execute(5000, 12, 5)
    assert isinstance(result, str)
    assert float(result) > 0


def test_tool_callable_dunder():
    result = calculate_sip_returns(5000, 12, 5)
    assert isinstance(result, str)


def test_to_schema_includes_parameters():
    schema = calculate_sip_returns.to_schema()
    assert "name" in schema
    assert "description" in schema
    assert "parameters" in schema
    params = schema["parameters"]
    assert "monthly_investment" in params
    assert "annual_rate" in params
    assert "years" in params
    assert params["monthly_investment"]["type"] == "float"
    assert params["years"]["type"] == "int"
    assert params["monthly_investment"]["required"] is True


def test_custom_basetool_subclass():
    stock_tool = StockPriceTool()
    assert stock_tool.name == "stock_price_fetcher"
    assert stock_tool.execute("TATA") == "₹980"
    assert "not found" in stock_tool.execute("ZOMATO")


def test_tool_repr():
    assert "<Tool:" in repr(calculate_sip_returns)


def test_tool_error_handling():
    result = calculate_sip_returns.execute("invalid", 12, 5)
    assert "Error" in result
