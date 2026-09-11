"""
tiffin_api.py
--------------
Production FastAPI Backend for Autonomous Tiffin Service Customer Care.
Exposes REST API endpoints for Vercel / Next.js / React Frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

from core.base_memory import SlidingWindowMemory
from core.base_llm import LLMFactory, TokenCostTracker, MockLLM
from agents.react_agent import ReActAgent
from agents.supervisor_agent import SupervisorAgent
from tools.tiffin_tools import (
    track_tiffin_delivery,
    pause_tiffin_subscription,
    get_daily_menu,
    issue_wallet_refund,
    MOCK_SUBSCRIPTIONS,
    MOCK_TODAYS_MENU
)

app = FastAPI(
    title="TiffinCare AI - Autonomous Customer Care API",
    description="Multi-Agent customer care backend for daily tiffin subscriptions.",
    version="1.0.0"
)

# Enable CORS for Vercel and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = "CUST101"
    provider: Optional[str] = "mock"
    api_key: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    customer_id: str
    tokens_used: int
    estimated_cost_inr: float
    updated_wallet_balance: Optional[float] = None
    updated_days_remaining: Optional[int] = None


def setup_tiffin_agents(provider: str, api_key: Optional[str] = None):
    """Sets up the specialist agents with tools and LLM."""
    if provider == "mock":
        res_llm = MockLLM(model_name="mock-tiffin-llm")
        # Pre-seed smart mock rules for tiffin requests
        res_llm.register_response("track", "THOUGHT: Customer wants delivery status.\nACTION: track_tiffin_delivery(CUST101)")
        res_llm.register_response("where", "THOUGHT: Customer asking for dabba location.\nACTION: track_tiffin_delivery(CUST101)")
        res_llm.register_response("pause", "THOUGHT: Customer wants to pause tiffin.\nACTION: pause_tiffin_subscription(CUST101, 2, Traveling)")
        res_llm.register_response("menu", "THOUGHT: Customer asking for today's menu.\nACTION: get_daily_menu(lunch)")
        res_llm.register_response("refund", "THOUGHT: Customer reporting spilled food. Crediting wallet.\nACTION: issue_wallet_refund(CUST101, 120, Spilled dal)")
        res_llm.register_response("spill", "THOUGHT: Customer reporting spilled food. Crediting wallet.\nACTION: issue_wallet_refund(CUST101, 120, Spilled dal)")
        
        # Responses after tool observation
        res_llm.register_response("out for delivery", "FINAL ANSWER: Your lunch dabba is Out for Delivery with Ramesh (Dabbawala #42)! ETA is ~12 minutes. His contact is +91 98200 12345.")
        res_llm.register_response("paused", "FINAL ANSWER: Done! Your tiffin delivery is paused for the next 2 days, and your subscription has been extended accordingly.")
        res_llm.register_response("paneer", "FINAL ANSWER: Today's lunch menu is: Paneer Butter Masala, Dal Tadka, 3 Phulkas, Jeera Rice, Salad & Gulab Jamun! 🍛")
        res_llm.register_response("credited", "FINAL ANSWER: I am so sorry about the spill! We have credited ₹120 instantly to your Tiffin Wallet. 🙏")
        llm = res_llm
    else:
        llm = LLMFactory.create(provider, api_key=api_key)

    # ReAct Support Agent with all Tiffin tools
    agent = ReActAgent(
        name="TiffinCareAgent",
        role="Empathetic Hospitality & Support Specialist for Indian Tiffin Service",
        system_prompt="Assist tiffin subscribers with delivery tracking, meal pausing, today's menu, and instant wallet refunds with warm Indian hospitality.",
        llm=llm,
        tools=[track_tiffin_delivery, pause_tiffin_subscription, get_daily_menu, issue_wallet_refund],
        memory=SlidingWindowMemory(max_messages=8),
        max_iterations=4
    )
    return agent


@app.get("/")
def root():
    return {
        "service": "TiffinCare Multi-Agent AI API",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/customer/{customer_id}")
def get_customer_profile(customer_id: str):
    cid = customer_id.upper().strip()
    cust = MOCK_SUBSCRIPTIONS.get(cid)
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"customer_id": cid, **cust}


@app.get("/api/menu")
def get_menu():
    return MOCK_TODAYS_MENU


@app.post("/api/chat", response_model=ChatResponse)
def handle_customer_chat(payload: ChatRequest):
    cid = (payload.customer_id or "CUST101").upper().strip()
    user_msg = payload.message.strip()

    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    agent = setup_tiffin_agents(payload.provider or "mock", payload.api_key)

    with TokenCostTracker(cost_per_1k_tokens_inr=0.20) as tracker:
        # Run agent through autonomous ReAct loop
        response_text = agent.run(f"[Customer {cid}]: {user_msg}")
        tracker.record_usage(user_msg, response_text)

    # Fetch updated customer state
    cust_data = MOCK_SUBSCRIPTIONS.get(cid, {})

    return ChatResponse(
        reply=response_text,
        customer_id=cid,
        tokens_used=tracker.total_tokens,
        estimated_cost_inr=tracker.total_cost_inr,
        updated_wallet_balance=cust_data.get("wallet_balance"),
        updated_days_remaining=cust_data.get("days_remaining")
    )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TiffinCare AI FastAPI Server on http://0.0.0.0:8000 ...")
    uvicorn.run("tiffin_api:app", host="0.0.0.0", port=8000, reload=True)
