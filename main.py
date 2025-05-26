import requests
import time
import os

# === Discord Webhook ===
WEBHOOK_URL = "https://discord.com/api/webhooks/1376351312312270888/_XJsB8JM7r4oCyipoIsrZfMw4j5peaLDNO9ZhHzJk9kEy0_TY-u9GJF2ZcmI1Pk2mGQU"

# === Scanner Settings ===
MAX_PRICE = 5.00
MIN_VOLUME = 50000
EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]
FINVIZ_URL = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers&f=sh_price_u5,sh_relvol_o1,sh_avgvol_o50"

# === Criteria Filter ===
def is_valid_stock(stock):
    try:
        price = float(stock["price"])
        volume = int(stock["volume"].replace(",", ""))
        exchange = stock.get("exchange", "")
        return (
            price <= MAX_PRICE and
            volume >= MIN_VOLUME and
            exchange in EXCHANGES
        )
    except:
        return False

# === Discord Message Sender ===
def send_alert(ticker, reason):
    chart_link = f"https://www.tradingview.com/chart/?symbol={ticker}"
    message = {
        "username": "Spike Scanner",
        "content": f"**{ticker}** triggered Phase 9 scanner\n**Reason:** {reason}\n[View on TradingView]({chart_link})"
    }
    requests.post(WEBHOOK_URL, json=message)

# === Phase 9 Gainer Feed ===
def fetch_top_gainers():
    try:
        response = requests.get("https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey=demo")
        return response.json()
    except:
        return []

# === Scanner Engine ===
def run_scanner():
    print("[SCAN STARTED]")
    tickers = fetch_top_gainers()
    qualified = []

    for stock in tickers:
        symbol = stock.get("symbol")
        if is_valid_stock(stock):
            qualified.append(symbol)
            send_alert(symbol, "Top Gainer + Under $5")

    if not qualified:
        print("[NO QUALIFIED SETUPS]")
    else:
        print("[ALERTS SENT]:", qualified)

# === Run Script ===
if __name__ == "__main__":
    run_scanner()
