"""
tiffin_server.py
-----------------
Zero-Dependency Autonomous Tiffin Customer Care HTTP REST Server.
Uses 100% Python standard library (http.server, json, urllib).
Provides CORS support and full JSON API for Vercel / React Frontend.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.parse
from typing import Optional

from core.base_memory import SlidingWindowMemory
from core.base_llm import LLMFactory, TokenCostTracker, MockLLM
from agents.react_agent import ReActAgent
from tools.tiffin_tools import (
    track_tiffin_delivery,
    pause_tiffin_subscription,
    get_daily_menu,
    issue_wallet_refund,
    MOCK_SUBSCRIPTIONS,
    MOCK_TODAYS_MENU
)


def setup_tiffin_agent(provider: str = "mock", api_key: Optional[str] = None) -> ReActAgent:
    """Sets up the ReAct tiffin support agent."""
    if provider == "mock":
        res_llm = MockLLM(model_name="mock-tiffin-llm")
        res_llm.register_response("track", "THOUGHT: Customer wants delivery status.\nACTION: track_tiffin_delivery(CUST101)")
        res_llm.register_response("where", "THOUGHT: Customer asking for dabba location.\nACTION: track_tiffin_delivery(CUST101)")
        res_llm.register_response("pause", "THOUGHT: Customer wants to pause tiffin.\nACTION: pause_tiffin_subscription(CUST101, 2, Traveling)")
        res_llm.register_response("menu", "THOUGHT: Customer asking for today's menu.\nACTION: get_daily_menu(lunch)")
        res_llm.register_response("refund", "THOUGHT: Customer reporting food issue. Crediting wallet.\nACTION: issue_wallet_refund(CUST101, 120, Spilled dal)")
        res_llm.register_response("spill", "THOUGHT: Customer reporting spilled food. Crediting wallet.\nACTION: issue_wallet_refund(CUST101, 120, Spilled dal)")
        
        # Tool observations -> Final response
        res_llm.register_response("out for delivery", "FINAL ANSWER: Your lunch dabba is Out for Delivery with Ramesh (Dabbawala #42)! ETA: ~12 mins. Contact: +91 98200 12345.")
        res_llm.register_response("paused", "FINAL ANSWER: Done! Your tiffin delivery is paused for the next 2 days and your subscription is extended by 2 days. ⏸️")
        res_llm.register_response("paneer", "FINAL ANSWER: Today's lunch menu is: Paneer Butter Masala, Dal Tadka, 3 Phulkas, Jeera Rice, Salad & Gulab Jamun! 🍛")
        res_llm.register_response("credited", "FINAL ANSWER: We apologize for the dal spill! ₹120 has been refunded instantly to your Tiffin Wallet. 🙏")
        llm = res_llm
    else:
        llm = LLMFactory.create(provider, api_key=api_key)

    return ReActAgent(
        name="TiffinCareAgent",
        role="Empathetic Tiffin Hospitality Specialist",
        system_prompt="Assist tiffin subscribers with delivery tracking, meal pausing, daily menus, and instant wallet refunds with warm Indian hospitality.",
        llm=llm,
        tools=[track_tiffin_delivery, pause_tiffin_subscription, get_daily_menu, issue_wallet_refund],
        memory=SlidingWindowMemory(max_messages=8),
        max_iterations=4
    )


class TiffinAPIHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/api/health":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"service": "TiffinCare AI API", "status": "healthy"}).encode("utf-8"))

        elif path.startswith("/api/customer/"):
            cid = path.split("/")[-1].upper()
            cust = MOCK_SUBSCRIPTIONS.get(cid)
            if cust:
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"customer_id": cid, **cust}).encode("utf-8"))
            else:
                self.send_response(404)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Customer '{cid}' not found"}).encode("utf-8"))

        elif path == "/api/menu":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(MOCK_TODAYS_MENU).encode("utf-8"))

        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON payload"}).encode("utf-8"))
                return

            cid = data.get("customer_id", "CUST101").upper().strip()
            user_msg = data.get("message", "").strip()
            provider = data.get("provider", "mock")
            api_key = data.get("api_key")

            if not user_msg:
                self.send_response(400)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Message cannot be empty"}).encode("utf-8"))
                return

            agent = setup_tiffin_agent(provider=provider, api_key=api_key)

            with TokenCostTracker(cost_per_1k_tokens_inr=0.20) as tracker:
                reply = agent.run(f"[Customer {cid}]: {user_msg}")
                tracker.record_usage(user_msg, reply)

            cust = MOCK_SUBSCRIPTIONS.get(cid, {})

            response_payload = {
                "reply": reply,
                "customer_id": cid,
                "tokens_used": tracker.total_tokens,
                "estimated_cost_inr": tracker.total_cost_inr,
                "updated_wallet_balance": cust.get("wallet_balance"),
                "updated_days_remaining": cust.get("days_remaining")
            }

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()


def run_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), TiffinAPIHandler)
    print(f"🍱 TiffinCare AI Zero-Dependency Server running on http://0.0.0.0:{port} ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TiffinCare server...")
        server.server_close()


if __name__ == "__main__":
    run_server(8000)
