
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_FRACTION = 0.7  # fraction of scanned_data used for the grid search


def run_time_machine(df, confidence_thresh, stop_loss_pct, take_profit_pct, max_hold_days):
    """Unchanged simulation mechanics -- this part was never the bug."""
    capital = 10000.0
    shares_held = 0.0
    buy_price = 0.0
    days_held = 0
    in_position = False

    total_trades = 0
    winning_trades = 0

    for _, row in df.iterrows():
        current_price = row["Close"]
        ai_pred = row["AI_Prediction"]
        ai_conf = row["AI_Confidence"]

        if not in_position:
            if ai_pred == "bullish" and ai_conf >= confidence_thresh:
                shares_held = capital / current_price
                buy_price = current_price
                capital = 0.0
                in_position = True
                days_held = 0
        else:
            days_held += 1
            current_pnl_pct = (current_price - buy_price) / buy_price
            sell_reason = None

            if current_pnl_pct <= stop_loss_pct:
                sell_reason = "STOP LOSS"
            elif current_pnl_pct >= take_profit_pct:
                sell_reason = "TAKE PROFIT"
            elif days_held >= max_hold_days:
                sell_reason = "TIME OUT"

            if sell_reason:
                capital = shares_held * current_price
                shares_held = 0.0
                in_position = False
                total_trades += 1
                if current_price > buy_price:
                    winning_trades += 1

    if in_position:
        capital = shares_held * df.iloc[-1]["Close"]

    return capital, total_trades, winning_trades


def grid_search(df, confidences, stop_losses, take_profits, max_days):
    """Search ONLY on the data passed in -- caller is responsible for passing
    just the SEARCH window, never the holdout."""
    best_profit = -float("inf")
    best_params = {}
    best_stats = {}

    for conf in confidences:
        for sl in stop_losses:
            for tp in take_profits:
                for days in max_days:
                    final_cash, trades, wins = run_time_machine(df, conf, sl, tp, days)
                    profit = final_cash - 10000.0

                    if profit > best_profit:
                        best_profit = profit
                        best_params = {"Confidence": conf, "StopLoss": sl, "TakeProfit": tp, "MaxDays": days}
                        best_stats = {"FinalCash": final_cash, "Trades": trades, "Wins": wins}

    return best_params, best_stats, best_profit


def optimize_strategy(ticker):
    filepath = os.path.join(BASE_DIR, "scanned_data", f"{ticker}_scanned.csv")
    if not os.path.exists(filepath):
        print(f"No scanned data for {ticker} -- run ai_scanner.py first.")
        return

    print("\n" + "=" * 60)
    print(f"WALK-FORWARD OPTIMIZATION FOR {ticker}")
    print("=" * 60)

    df = pd.read_csv(filepath)
    n = len(df)
    split_idx = int(n * SEARCH_FRACTION)
    search_df = df.iloc[:split_idx]
    holdout_df = df.iloc[split_idx:]

    print(f"Scanned data: {n} rows (already restricted to the model's held-out TEST period)")
    print(f"  SEARCH window:  rows 0:{split_idx}   ({len(search_df)} rows) -- grid search runs here")
    print(f"  HOLDOUT window: rows {split_idx}:{n}  ({len(holdout_df)} rows) -- final number reported here\n")

    confidences = [50, 60, 70, 80]
    stop_losses = [-0.02, -0.04, -0.06]
    take_profits = [0.04, 0.08, 0.12]
    max_days = [5, 10, 20]
    total_combos = len(confidences) * len(stop_losses) * len(take_profits) * len(max_days)

    print(f"Running grid search over {total_combos} parameter combinations on SEARCH window only...")
    best_params, search_stats, search_profit = grid_search(
        search_df, confidences, stop_losses, take_profits, max_days
    )

    if not best_params:
        print(f"No profitable strategy found for {ticker} on the search window.")
        return

    search_win_rate = (search_stats["Wins"] / search_stats["Trades"] * 100) if search_stats["Trades"] > 0 else 0
    print(f"\nBest parameters found on SEARCH window:")
    print(f"  Confidence >= {best_params['Confidence']}%, "
          f"StopLoss {best_params['StopLoss']*100:.0f}%, "
          f"TakeProfit +{best_params['TakeProfit']*100:.0f}%, "
          f"MaxHold {best_params['MaxDays']}d")
    print(f"  SEARCH window result: +${search_profit:,.2f} profit, "
          f"{search_win_rate:.1f}% win rate ({search_stats['Wins']}/{search_stats['Trades']} trades)")

    # The number that matters: apply the winning, FIXED parameters to the
    # holdout window, which the grid search never touched.
    holdout_cash, holdout_trades, holdout_wins = run_time_machine(
        holdout_df,
        best_params["Confidence"], best_params["StopLoss"],
        best_params["TakeProfit"], best_params["MaxDays"],
    )
    holdout_profit = holdout_cash - 10000.0
    holdout_win_rate = (holdout_wins / holdout_trades * 100) if holdout_trades > 0 else 0

    print(f"\nHOLDOUT window result (same fixed parameters, unseen data):")
    print(f"  +${holdout_profit:,.2f} profit, {holdout_win_rate:.1f}% win rate "
          f"({holdout_wins}/{holdout_trades} trades)")

    if holdout_trades == 0:
        print("\n  No trades triggered on the holdout window with these parameters --")
        print("  not enough evidence to claim this strategy generalizes.")
    elif holdout_profit <= 0 <= search_profit:
        print("\n  WARNING: profitable on the search window but not on holdout.")
        print("  This is the signature of overfitting to the search window -- the")
        print("  parameters were likely fit to noise, not a real, repeatable edge.")
    else:
        degradation = (search_profit - holdout_profit) / max(abs(search_profit), 1e-9) * 100
        print(f"\n  Performance degradation from search -> holdout: {degradation:.1f}%")
        print("  Some degradation is normal and expected; a strategy whose holdout")
        print("  performance is reasonably close to its search performance has more")
        print("  credible evidence of a real edge than one that doesn't.")


if __name__ == "__main__":
    optimize_strategy("AAPL")
    optimize_strategy("TSLA")
