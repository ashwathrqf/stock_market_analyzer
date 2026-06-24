# Deep Quant — AI Market Pattern Scanner

An end-to-end pipeline that trains a ResNet50 vision model to read 30-day
candlestick charts and classify the next 10 days of price action as
**bullish**, **bearish**, or **neutral** — then backtests that model's
predictions as a simple trading strategy, with a Streamlit dashboard for
both live inference and interactive backtesting.

## Why this version exists

An earlier version of this pipeline trained and evaluated on overlapping
historical data — the backtest numbers it produced were not genuinely
out-of-sample, and a grid search over trading parameters was scored on the
same window it searched. Both are classic ways to make a strategy look
better than it is.

This version fixes that with a **strict, chronological train/test split**
applied at the very first step of the pipeline (see `pipeline_engine.py`),
so every later script — training, scanning, backtesting — operates on data
the model has genuinely never seen during training. The backtester also
does a **walk-forward split of its own**: parameters are searched on one
window and reported on a separate holdout window the search never touched.
See the "Methodology" section below for the full explanation.

## Pipeline overview

```
data_downloader.py   -> raw_numeric_data/{TICKER}.csv          (5y OHLCV per ticker)
pipeline_engine.py   -> dataset/train/{label}/*.png            (time-based split, see below)
                         dataset/test/{label}/*.png
                         dataset/split_log.json                 (per-ticker train/test date ranges)
model_training.py    -> financial_resnet.pth                   (trained on train/, validated on test/)
ai_scanner.py         -> scanned_data/{TICKER}_scanned.csv      (model predictions on TEST period only)
backtester.py         -> console output                        (walk-forward search + holdout report)
live_predictor.py     -> console output                        (live inference on today's data)
app.py                 -> Streamlit dashboard                   (live inference + interactive backtest UI)
```

Run them in that order the first time. After that, you can re-run any
individual stage independently as long as its inputs already exist.

## Methodology — the train/test split, explained

`pipeline_engine.py` takes each ticker's ~5 years of daily price history
and splits it **chronologically**, per ticker:

```
|------------------ TRAIN (first 75%) ------------------|--- TEST (last 25%) ---|
earliest date                                    split date              latest date
```

- Every candlestick image used to **train** the model comes only from the
  TRAIN period.
- Every candlestick image used to **validate** the model each epoch, and
  every day scanned by `ai_scanner.py` for the backtest, comes only from
  the TEST period — which is strictly later in time and was never shown to
  the model during training.
- The exact date ranges used for every ticker are written to
  `dataset/split_log.json` so the split is auditable, not just asserted.

On top of that, `backtester.py` does a **second, independent split** of
its own: it takes the (already out-of-sample) scanned data for a ticker
and splits *that* into a SEARCH window (where the stop-loss / take-profit
/ confidence-threshold grid search runs) and a HOLDOUT window (where the
winning, now-fixed parameters are evaluated once, with no further
tuning). This is the standard walk-forward validation pattern used to
catch overfitting in a parameter search — if a strategy's holdout
performance is much worse than its search-window performance, that's a
sign the "best" parameters were fit to noise rather than a real,
repeatable pattern. `backtester.py` prints both numbers explicitly so that
gap is visible.

**What this does and doesn't prove:** even with this split, a single
ticker's holdout window is a small sample, and "candlestick image → CNN →
trade signal" is a genuinely hard problem that most published research is
skeptical actually works reliably out of sample. The honest framing for
this project is "a correctly-validated experiment in vision-based price
pattern classification," not "a proven trading strategy." The dashboard
and console output are written to reflect that.

## Setup

**Hardware note:** this was built and tested for an NVIDIA GPU (RTX 4050
or similar). Training will run on CPU if no GPU is found, but a CNN over
tens of thousands of images will be dramatically slower — expect it to be
impractical on CPU alone.

```bash
git clone <your-repo-url>
cd deep-quant
python -m venv .venv
```

Activate it:
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

**Install PyTorch with CUDA support first, separately** — the default
`pip install torch` often grabs a CPU-only build, which won't use your
GPU. Check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
for the exact command for your CUDA version, or try:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install everything else:
```bash
pip install -r requirements.txt
```

Verify the GPU is visible to PyTorch before training:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU found')"
```
This should print `True` and your GPU's name. If it prints `False`,
training will still run but will be CPU-bound — fix the PyTorch/CUDA
install before proceeding to avoid a very long training run.

## Running the pipeline, step by step

Run these from the `project/` folder, in order, the first time:

### 1. Download the data
```bash
python data_downloader.py
```
Scrapes the current S&P 500 ticker list and downloads 5 years of daily
OHLCV data per ticker into `raw_numeric_data/`. Takes a few minutes
depending on connection speed (500 tickers, 20 parallel threads).

### 2. Generate the labeled image dataset
```bash
python pipeline_engine.py
```
Slices each ticker's history into 30-day candlestick images, labels each
one by what happened in the following 10 days, and — critically — splits
each ticker's timeline into TRAIN and TEST *before* generating any images.
Writes `dataset/train/`, `dataset/test/`, and `dataset/split_log.json`.
This generates tens of thousands of small PNG images and will take a
while; it's CPU-bound (image rendering), not GPU-bound.

### 3. Train the model
```bash
python model_training.py
```
Trains a ResNet50 (ImageNet-pretrained, with `layer4` and the final
classifier head unfrozen) on `dataset/train/`, and after every epoch runs
a real held-out validation pass on `dataset/test/` in `model.eval()` mode.
Saves `financial_resnet.pth` whenever validation accuracy improves — not
training accuracy, which is what should be quoted anywhere this project
is described. Default is 30 epochs; edit `TOTAL_EPOCHS` in the script if
you want more or fewer. On an RTX 4050 this should run at a reasonable
pace depending on dataset size; watch the printed VAL accuracy, not the
train accuracy, to judge whether it's working.

### 4. Scan the held-out period with the trained model
```bash
python ai_scanner.py
```
Runs the trained model day-by-day over each ticker's TEST period only
(read from `dataset/split_log.json`) and saves predictions + confidence
to `scanned_data/{TICKER}_scanned.csv`. Default watchlist is AAPL, TSLA,
SPY — edit `test_watchlist` in the script to change it.

### 5. Backtest with walk-forward validation
```bash
python backtester.py
```
Splits each ticker's scanned data into a SEARCH window and a HOLDOUT
window, grid-searches stop-loss / take-profit / confidence-threshold /
max-hold-time on the SEARCH window only, then reports the winning,
now-fixed parameters' performance on the untouched HOLDOUT window. Read
both numbers — a strategy that only looks good on the search window and
falls apart on holdout is telling you something important.

### 6. (Optional) Run live inference from the command line
```bash
python live_predictor.py
```
Pulls today's live chart for a watchlist of tickers and prints the
model's forecast for each. Pure inference, no training or evaluation
logic — a quick sanity check that the model loads and runs.

### 7. Launch the dashboard
```bash
streamlit run app.py
```
Opens a two-page Streamlit app:
- **Live Oracle** — type a ticker, get a live chart and the model's
  forecast.
- **Interactive Backtester** — load a scanned ticker's held-out test
  period and experiment with stop-loss / take-profit / confidence
  sliders interactively, with a clear note that this single-window result
  is exploratory, not a substitute for the walk-forward check in
  `backtester.py`.

## Project structure

```
project/
├── data_downloader.py     # Step 1: download raw OHLCV data
├── pipeline_engine.py     # Step 2: build labeled, time-split image dataset
├── model_training.py      # Step 3: train + validate the CNN
├── ai_scanner.py           # Step 4: scan held-out period with trained model
├── backtester.py           # Step 5: walk-forward grid search + holdout report
├── live_predictor.py       # Step 6 (optional): CLI live inference
├── app.py                   # Step 7: Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

Generated at runtime (gitignored, not committed):
```
raw_numeric_data/    # from data_downloader.py
dataset/              # from pipeline_engine.py (train/, test/, split_log.json)
scanned_data/          # from ai_scanner.py
financial_resnet.pth   # from model_training.py
```

## Tech stack

Python, PyTorch, torchvision (ResNet50 transfer learning), yfinance,
mplfinance, pandas, NumPy, Streamlit.

## Known limitations

- A 75/25 time-based split per ticker is simple and auditable, but with
  only ~5 years of data per ticker, the TEST period for any one ticker is
  still fairly short (roughly a year). Results on any single ticker's
  holdout should be read as a small sample, not a statistically strong
  claim.
- The label threshold (3% move to count as bullish/bearish) and window
  sizes (30-day input, 10-day lookahead) were chosen as reasonable
  defaults, not tuned via a separate validation process — they're a
  reasonable starting point for further experimentation, not a final
  answer.
- This project does not account for realistic transaction costs, slippage,
  or market impact in the backtester — `run_time_machine()` assumes
  instant fills at the closing price with no friction.
