import requests
import os
from datetime import datetime
from gainer_feed import get_top_gainers
from alert import send_discord_alert

# === CONFIG ===
UNDER_PRICE = 5.00
MIN_VOLUME = 50000

# === PHASE 9 ===
def scan():
    print(f"[SCAN STARTED] {datetime.now().strftime('%H:%M:%S')}")
    tickers = get_top_gainers()
    print(f"[Phase 9] Loaded {len(tickers)} tickers: {tickers}")

    qualified = []
    for ticker in tickers:
        try:
            data = requests.get(f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}").json()
            quote = data['quoteResponse']['result'][0]
            price = quote.get('regularMarketPrice', 0)
            volume = quote.get('regularMarketVolume', 0)

            if price <= UNDER_PRICE and volume >= MIN_VOLUME:
                qualified.append(ticker)
        except Exception as e:
            print(f"Error checking {ticker}: {e}")

    if qualified:
        print(f"[QUALIFIED SETUPS] {qualified}")
        send_discord_alert(qualified)
    else:
        print("[NO QUALIFIED SETUPS]")

if __name__ == '__main__':
    scan()
