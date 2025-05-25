import requests

WEBHOOK_URL = "https://discord.com/api/webhooks/1376195990331195553/ozSWULbEVcN4-fgN8n8MJRAIO7LblqFrCb5O_7AB-PF9spG_NimZz-XoZdVOeYYTcYs_"

def send_alert(message):
    data = {
        "content": f"**[PREMARKET ALERT]** {message}"
    }

    try:
        requests.post(WEBHOOK_URL, json=data)
        print(f"[ALERT SENT] {message}")
    except Exception as e:
        print(f"[ERROR] Failed to send alert: {e}")
