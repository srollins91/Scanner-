import requests

WEBHOOK_URL = "https://discord.com/api/webhooks/1376351312312270888/_XJsB8JM7r4oCyipoIsrZfMw4j5peaLDNO9ZhHzJk9kEy0_TY-u9GJF2ZcmI1Pk2mGQU"

def send_discord_alert(tickers):
    message = {
        "content": f"**Qualified Pre-Market Setups:** {', '.join(tickers)}"
    }
    requests.post(WEBHOOK_URL, json=message)
