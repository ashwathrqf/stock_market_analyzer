
import io
import json
import os

import mplfinance as mpf
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

WINDOW_SIZE = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRAIN_FILE = os.path.join(BASE_DIR, "financial_resnet.pth")
SPLIT_LOG_PATH = os.path.join(BASE_DIR, "dataset", "split_log.json")


def build_model(num_classes=3):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


def load_ai(model_path=BRAIN_FILE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device


def _load_clean_csv(filepath):
    df = pd.read_csv(filepath, skiprows=2)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def run_ai_scanner(target_tickers):
    if not os.path.exists(SPLIT_LOG_PATH):
        print(f"{SPLIT_LOG_PATH} not found. Run pipeline_engine.py first -- it's what")
        print("defines each ticker's held-out TEST period, which is what this scanner")
        print("restricts itself to. Scanning without it would silently include")
        print("data the model trained on, which defeats the point of backtesting.")
        return

    with open(SPLIT_LOG_PATH) as f:
        split_log = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    categories = ["bearish", "bullish", "neutral"]

    print("Loading trained model...")
    model, device = load_ai()

    scanned_dir = os.path.join(BASE_DIR, "scanned_data")
    os.makedirs(scanned_dir, exist_ok=True)

    mc = mpf.make_marketcolors(up="g", down="r", edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle="", facecolor="black",
                                edgecolor="black", figcolor="black")

    for ticker in target_tickers:
        filepath = os.path.join(BASE_DIR, "raw_numeric_data", f"{ticker}.csv")
        if not os.path.exists(filepath):
            print(f"Could not find {ticker}.csv, skipping...")
            continue
        if ticker not in split_log or split_log[ticker]["test_start"] is None:
            print(f"No recorded TEST period for {ticker} in split_log.json, skipping...")
            continue

        test_start = pd.Timestamp(split_log[ticker]["test_start"])
        print(f"\nScanning {ticker} -- restricting to TEST period starting {test_start.date()} "
              f"(never seen during training)...")

        df = _load_clean_csv(filepath)
        df = df[df.index >= test_start]

        if len(df) < WINDOW_SIZE + 1:
            print(f"  Not enough test-period data for {ticker} ({len(df)} rows), skipping...")
            continue

        df["AI_Prediction"] = "none"
        df["AI_Confidence"] = 0.0

        total_days = len(df)
        for i in range(WINDOW_SIZE, total_days):
            chart_window = df.iloc[i - WINDOW_SIZE: i]

            buf = io.BytesIO()
            mpf.plot(
                chart_window, type="candle", style=style, axisoff=True,
                savefig=dict(fname=buf, format="png", dpi=50, bbox_inches="tight", pad_inches=0),
                closefig=True,
            )
            buf.seek(0)

            raw_image = Image.open(buf).convert("RGB")
            input_tensor = transform(raw_image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)

            prediction = categories[predicted_idx.item()]
            confidence_pct = confidence.item() * 100

            df.at[df.index[i], "AI_Prediction"] = prediction
            df.at[df.index[i], "AI_Confidence"] = confidence_pct

            if i % 100 == 0:
                print(f"  Processed {i}/{total_days} test-period days...")

        output_path = os.path.join(scanned_dir, f"{ticker}_scanned.csv")
        df.to_csv(output_path)
        print(f"Finished scanning {ticker} (test-period only). Saved to {output_path}")


if __name__ == "__main__":
    test_watchlist = ["AAPL", "TSLA", "SPY"]
    run_ai_scanner(test_watchlist)
