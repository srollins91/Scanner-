from alert import send_alert

# Example inside your scan loop:
if is_valid_setup(ticker_data):  # your custom logic
    print(f"[ALERT] {ticker}")
    send_alert(ticker, reason="A+ Spike Setup")
