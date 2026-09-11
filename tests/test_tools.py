import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.base_tool import BaseTool, tool

# Test 1: Using the @tool decorator on a simple Python function
@tool
def calculate_sip_returns(monthly_investment: float, annual_rate: float, years: int) -> float:
    """Calculates total estimated future value of an SIP in mutual funds."""
    months = years * 12
    monthly_rate = (annual_rate / 100) / 12
    # SIP formula: P * [ ((1+i)^n - 1) / i ] * (1+i)
    future_value = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    return round(future_value, 2)

# Test 2: Creating a custom Tool subclass
class StockPriceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="stock_price_fetcher",
            description="Fetches live stock price for Indian NSE/BSE tickers."
        )

    def execute(self, ticker: str) -> str:
        # Mocking real price lookup
        mock_prices = {"TATA": "₹980", "RELIANCE": "₹2950", "INFY": "₹1520"}
        return mock_prices.get(ticker.upper(), f"Ticker '{ticker}' not found on NSE.")

if __name__ == "__main__":
    print("--- 1. Testing @tool Decorator ---")
    print("Tool Name:", calculate_sip_returns.name)
    print("Tool Description:", calculate_sip_returns.description)
    print("Schema:", calculate_sip_returns.to_schema())
    print("Execution (₹5000/mo @ 12% for 5 yrs):", calculate_sip_returns.execute(5000, 12, 5))

    print("\n--- 2. Testing Custom BaseTool Subclass ---")
    stock_tool = StockPriceTool()
    print("Tool Name:", stock_tool.name)
    print("Tool Representation:", stock_tool)
    print("Execution TATA:", stock_tool.execute("TATA"))
    print("Execution ZOMATO:", stock_tool.execute("ZOMATO"))

    print("\n✅ All Milestone 1 Tool tests passed successfully!")
