import requests
import os

def send_alert(ticker):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    tradingview_link = f"https://www.tradingview.com/chart/?symbol=NASDAQ:{ticker.upper()}"
    message = f"**ALERT: {ticker.upper()}**\nReason: A+ Setup\n{tradingview_link}"

    if webhook_url:
        requests.post(webhook_url, json={"content": message})
    else:
        print("Webhook URL not set.")
