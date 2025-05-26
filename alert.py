import requests

WEBHOOK_URL = "https://discord.com/api/webhooks/1376351312312270888/_XJsB8JM7r4oCyipoIsrZfMw4j5peaLDNO9ZhHzJk9kEy0_TY-u9GJF2ZcmI1Pk2mGQU"

def send_alert(ticker, reason="A+ Setup"):
    tradingview_url = f"https://www.tradingview.com/symbols/{ticker.upper()}/"
    message = {
        "username": "Spike Scanner",
        "embeds": [
            {
                "title": f"{ticker.upper()} Triggered",
                "description": f"**Reason:** {reason}\n[Open in TradingView]({tradingview_url})",
                "color": 3066993
            }
        ]
    }
    try:
        response = requests.post(WEBHOOK_URL, json=message)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ALERT ERROR] {e}")
