import glob
import json
import os

import mplfinance as mpf
import pandas as pd

TRAIN_FRACTION = 0.75

WINDOW_SIZE = 30   # days of price history shown to the model per image
FUTURE_SIZE = 10   # days looked ahead to assign a label
STRIDE = 10        # step size when sliding the window across history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_numeric_data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def calculate_trend(future_window):
    """Labeling logic, unchanged from the original -- this part was correct."""
    if future_window.empty or len(future_window) < 5:
        return None

    threshold = 0.03
    baseline_price = future_window["Open"].iloc[0]

    max_gain = (future_window["High"].max() - baseline_price) / baseline_price
    max_loss = (future_window["Low"].min() - baseline_price) / baseline_price

    if max_gain >= threshold and abs(max_loss) < threshold:
        return "bullish"
    elif abs(max_loss) >= threshold and max_gain < threshold:
        return "bearish"
    else:
        return "neutral"


def _load_clean_csv(filepath):
    df = pd.read_csv(filepath, skiprows=2)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def _generate_images_for_range(df, ticker, mpf_style, out_root, total_counter):
    """Slide a window across df and save labeled candlestick images under out_root."""
    n = len(df)
    for i in range(0, n - (WINDOW_SIZE + FUTURE_SIZE), STRIDE):
        chart_window = df.iloc[i: i + WINDOW_SIZE]
        future_window = df.iloc[i + WINDOW_SIZE: i + WINDOW_SIZE + FUTURE_SIZE]

        label = calculate_trend(future_window)
        if label is None:
            continue

        filename = f"{out_root}/{label}/{ticker}_{i}.png"
        mpf.plot(
            chart_window, type="candle", style=mpf_style, axisoff=True,
            savefig=dict(fname=filename, dpi=50, bbox_inches="tight", pad_inches=0),
            closefig=True,
        )
        total_counter[0] += 1
        if total_counter[0] % 500 == 0:
            print(f"  Generated {total_counter[0]} images so far...")


def generate_mass_dataset():
    for split in ["train", "test"]:
        for label in ["bullish", "bearish", "neutral"]:
            os.makedirs(os.path.join(DATASET_DIR, split, label), exist_ok=True)

    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    print(f"Found {len(csv_files)} stock files. Starting mass generation...")
    print(f"Per-ticker split: first {TRAIN_FRACTION:.0%} of history -> TRAIN, "
          f"remaining {1 - TRAIN_FRACTION:.0%} -> TEST (strictly later in time).\n")

    mc = mpf.make_marketcolors(up="g", down="r", edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle="", facecolor="black",
                                edgecolor="black", figcolor="black")

    train_counter = [0]
    test_counter = [0]
    split_log = {}

    for filepath in csv_files:
        ticker = os.path.basename(filepath).replace(".csv", "")
        try:
            df = _load_clean_csv(filepath)
            if df.empty or len(df) < 100:
                continue

            split_idx = int(len(df) * TRAIN_FRACTION)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            split_log[ticker] = {
                "train_start": str(train_df.index.min().date()),
                "train_end": str(train_df.index.max().date()),
                "test_start": str(test_df.index.min().date()) if not test_df.empty else None,
                "test_end": str(test_df.index.max().date()) if not test_df.empty else None,
            }

            if len(train_df) >= WINDOW_SIZE + FUTURE_SIZE:
                _generate_images_for_range(
                    train_df, ticker, style, os.path.join(DATASET_DIR, "train"), train_counter
                )
            if len(test_df) >= WINDOW_SIZE + FUTURE_SIZE:
                _generate_images_for_range(
                    test_df, ticker, style, os.path.join(DATASET_DIR, "test"), test_counter
                )

        except Exception as e:
            print(f"  Skipped {ticker}: {e}")
            continue

    # Save the exact per-ticker split dates so anyone (including future you)
    # can verify there's no overlap between train and test periods.
    split_log_path = os.path.join(DATASET_DIR, "split_log.json")
    with open(split_log_path, "w") as f:
        json.dump(split_log, f, indent=2)

    print(f"\nMass generation complete.")
    print(f"  TRAIN images: {train_counter[0]}  -> dataset/train/")
    print(f"  TEST images:  {test_counter[0]}  -> dataset/test/")
    print(f"  Per-ticker date ranges written to {split_log_path}")
    print(
        "\nBecause TEST images come from a strictly later time period per ticker, "
        "the model trained on dataset/train/ has never seen any pattern from "
        "dataset/test/ -- so accuracy measured there is genuinely out-of-sample."
    )


if __name__ == "__main__":
    generate_mass_dataset()
