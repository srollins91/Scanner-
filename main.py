import time
from gainer_feed import get_top_gainers
from alert import send_alert
import random

def is_spike_candidate(ticker):
    # TEMP PLACEHOLDER — replace with real logic later
    return random.random() < 0.1  # Simulate 10% chance

while True:
    print("[SCAN STARTED]")
    gainers = get_top_gainers()
    print(f"[Phase 9] Loaded {len(gainers)} tickers: {gainers}")

    found = False
    for ticker in gainers:
        if is_spike_candidate(ticker):
            send_alert(f"{ticker} is triggering spike scan setup!")
            found = True

    if not found:
        print("[NO QUALIFIED SETUPS]")

    print("[SLEEPING 5 MINUTES]")
    time.sleep(300)
