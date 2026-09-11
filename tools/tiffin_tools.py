"""
tools/tiffin_tools.py
----------------------
Production Tools for Autonomous Tiffin Service Customer Care.
"""

from core.base_tool import tool
from typing import Dict, Any
import datetime

# Mock Customer & Subscription Database
MOCK_SUBSCRIPTIONS = {
    "CUST101": {
        "name": "Kunal Sharma",
        "plan": "Monthly Premium (Lunch + Dinner)",
        "status": "Active",
        "days_remaining": 18,
        "wallet_balance": 240.0,
        "diet": "Standard North Indian (Non-Veg Wed/Fri)",
        "address": "B-402, Sunshine Heights, Andheri West, Mumbai"
    },
    "CUST102": {
        "name": "Pooja Patel",
        "plan": "Lunch Only (Pure Veg/Jain)",
        "status": "Active",
        "days_remaining": 12,
        "wallet_balance": 50.0,
        "diet": "Strict Jain (No Onion, No Garlic)",
        "address": "Flat 12, Gokul Dham, Borivali East, Mumbai"
    }
}

MOCK_TODAYS_MENU = {
    "lunch": "Paneer Butter Masala, Dal Tadka, 3 Phulkas, Jeera Rice, Fresh Salad, Gulab Jamun.",
    "dinner": "Aloo Gobi Matar, Panchmel Dal, 3 Butter Roti, Steamed Rice, Boondi Raita."
}

MOCK_DELIVERIES = {
    "CUST101": {
        "status": "Out for Delivery",
        "rider_name": "Ramesh (Dabbawala #42)",
        "rider_phone": "+91 98200 12345",
        "eta_minutes": 12,
        "location": "Near Andheri Station signal"
    },
    "CUST102": {
        "status": "Delivered",
        "delivered_at": "12:45 PM",
        "received_by": "Security Desk"
    }
}


@tool
def track_tiffin_delivery(customer_id: str) -> str:
    """Checks the real-time delivery status and ETA of today's tiffin dabba for a customer ID."""
    cid = customer_id.upper().strip()
    delivery = MOCK_DELIVERIES.get(cid)
    if not delivery:
        return f"No active delivery found for Customer ID '{cid}'."
    
    if delivery["status"] == "Out for Delivery":
        return f"🛵 Status: Out for Delivery with {delivery['rider_name']}. ETA: ~{delivery['eta_minutes']} mins (Current location: {delivery['location']}). Rider Contact: {delivery['rider_phone']}."
    else:
        return f"✅ Status: Delivered at {delivery.get('delivered_at', 'afternoon')} (Handed to: {delivery.get('received_by', 'Customer')})."


@tool
def pause_tiffin_subscription(customer_id: str, days_to_pause: int, reason: str = "Traveling") -> str:
    """Pauses daily tiffin meal delivery for specified number of days and extends the subscription end date."""
    cid = customer_id.upper().strip()
    cust = MOCK_SUBSCRIPTIONS.get(cid)
    if not cust:
        return f"Customer ID '{cid}' not found in database."
    
    days = int(days_to_pause)
    cust["days_remaining"] += days
    return f"⏸️ Successfully PAUSED tiffin for {days} day(s) (Reason: {reason}). Your subscription has been extended by {days} days. New remaining days: {cust['days_remaining']}."


@tool
def get_daily_menu(meal_type: str = "lunch") -> str:
    """Returns today's fresh cooked menu for 'lunch' or 'dinner'."""
    m_type = meal_type.lower().strip()
    menu = MOCK_TODAYS_MENU.get(m_type)
    if not menu:
        return f"Menu for '{meal_type}' is being prepared. Today's Lunch: {MOCK_TODAYS_MENU['lunch']} | Dinner: {MOCK_TODAYS_MENU['dinner']}"
    return f"🍛 Today's {m_type.capitalize()} Menu: {menu}"


@tool
def issue_wallet_refund(customer_id: str, amount_inr: float, issue_reason: str) -> str:
    """Issues instant refund credit to customer's in-app Tiffin Wallet for late or damaged meals."""
    cid = customer_id.upper().strip()
    cust = MOCK_SUBSCRIPTIONS.get(cid)
    if not cust:
        return f"Customer ID '{cid}' not found in database."
    
    amt = float(amount_inr)
    cust["wallet_balance"] += amt
    return f"💰 Refund of ₹{amt:.2f} credited instantly to your Tiffin Wallet for issue: '{issue_reason}'. Updated Wallet Balance: ₹{cust['wallet_balance']:.2f}."
