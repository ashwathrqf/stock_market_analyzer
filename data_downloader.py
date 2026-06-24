import os
import io
import concurrent.futures

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_numeric_data")


def get_sp500_tickers():
    print("Scraping live S&P 500 ticker list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})[0]

    return [t.replace(".", "-") for t in table["Symbol"].tolist()]


def download_single_stock(ticker):
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if not df.empty and len(df) > 50:
            df.to_csv(os.path.join(RAW_DATA_DIR, f"{ticker}.csv"))
            return True
    except Exception:
        pass
    return False


def mass_download():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    tickers = get_sp500_tickers()

    print(f"Firing up 20 parallel threads to download {len(tickers)} stocks...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(download_single_stock, tickers))

    print(f"Done. Downloaded {sum(results)} / {len(tickers)} CSVs to raw_numeric_data/.")
    print(
        "Next step: run pipeline_engine.py, which will split each ticker's "
        "5-year history into a TRAIN period and a strictly later, non-overlapping "
        "TEST period before generating any candlestick images."
    )


if __name__ == "__main__":
    mass_download()
