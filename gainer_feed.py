import requests
from bs4 import BeautifulSoup

def get_top_gainers():
    url = "https://finance.yahoo.com/gainers"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    tickers = []
    for row in soup.select("table tbody tr"):
        cols = row.find_all("td")
        if len(cols) > 0:
            ticker = cols[0].text.strip()
            tickers.append(ticker)

    return tickers
