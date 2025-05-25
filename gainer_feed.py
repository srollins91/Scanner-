import requests
from bs4 import BeautifulSoup

def get_top_gainers():
    url = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find_all("a", class_="screener-link-primary")
        tickers = [t.text.strip() for t in table]
        return tickers[:15]  # Limit to top 15 for speed
    except Exception as e:
        print(f"[ERROR] Failed to fetch gainers: {e}")
        return []
