import time
from gainer_feed import get_top_gainers

def is_valid_setup(ticker):
    # Placeholder logic for now
    return "A" in ticker

while True:
    print("[SCAN STARTED]", time.strftime("%H:%M:%S"))
    tickers = get_top_gainers()
    print(f"[Phase 9] Loaded {len(tickers)} tickers: {tickers}")

    qualified = [t for t in tickers if is_valid_setup(t)]
    if qualified:
        print(f"[ALERT] Qualified setups: {qualified}")
    else:
        print("[NO QUALIFIED SETUPS]")

    print("[SLEEPING 5 MINUTES]")
    time.sleep(300)
