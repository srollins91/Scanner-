import requests

def send_discord_alert(tickers):
    webhook_url = "https://discord.com/api/webhooks/1376351312312270888/_XJsB8JM7r4oCyipoIsrZfMw4j5peaLDNO9ZhHzJk9kEy0_TY-u9GJF2ZcmI1Pk2mGQU"
    if not tickers:
        return
    content = f"**Qualified Stocks:** {', '.join(tickers)}"
    data = {"content": content}
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()
        print(f"[ALERT SENT] {content}")
    except Exception as e:
        print(f"[ALERT FAILED] {e}")
