

import io
import os

import mplfinance as mpf
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import yfinance as yf
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "financial_resnet.pth")


def build_model(num_classes=3):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


def load_ai(model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device


def get_live_chart_image(ticker):
    print(f"Fetching live market data for {ticker}...")
    stock = yf.Ticker(ticker)
    data = stock.history(period="45d")

    if data.empty:
        print(f"ERROR: Yahoo Finance returned no data for {ticker}.")
        return None

    data = data.dropna()
    for col in ["Open", "High", "Low", "Close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    chunk = data.tail(30)

    mc = mpf.make_marketcolors(up="g", down="r", edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle="", facecolor="black",
                                edgecolor="black", figcolor="black")

    buf = io.BytesIO()
    mpf.plot(chunk, type="candle", style=style, axisoff=True,
              savefig=dict(fname=buf, format="png", dpi=50, bbox_inches="tight", pad_inches=0))
    buf.seek(0)

    return Image.open(buf).convert("RGB")


def predict_trend(ticker, model, device):
    categories = ["bearish", "bullish", "neutral"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_image = get_live_chart_image(ticker)
    if raw_image is None:
        return

    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    print("Analyzing visual pattern...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    prediction = categories[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    print("\n" + "=" * 40)
    print(f"  LIVE PREDICTION FOR {ticker.upper()}")
    print("=" * 40)
    print(f"  Forecast   : {prediction.upper()}")
    print(f"  Confidence : {confidence_pct:.2f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    important_tickers = [
        "NVDA", "TSLA", "META",
        "SPY", "QQQ",
        "PLTR", "COIN", "AMD",
        "KO", "JNJ", "MCD",
    ]

    print("=" * 50)
    print("LIVE MARKET SWEEP")
    print("=" * 50 + "\n")

    model, device = load_ai()

    for ticker in important_tickers:
        try:
            predict_trend(ticker, model, device)
        except Exception as e:
            print(f"Skipped {ticker} due to a live data error: {e}")
            continue
