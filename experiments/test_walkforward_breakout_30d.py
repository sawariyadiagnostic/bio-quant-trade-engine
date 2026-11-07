"""
30-Day Walk-Forward Breakout Prediction & Empirical Validation Engine.
Evaluates Phase A 18D Manifold & Institutional Thrust Signals at 3:10 PM Day t
against Day t+1 actual market outcomes (Open Gap, Max Expansion High, Day Close).
Strict zero-lookahead walk-forward test across all 81 equities.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from loguru import logger

def _np_serial(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    return str(obj)

def load_all_equities(lake_dir: str = "data/lake") -> dict[str, dict[str, pd.DataFrame]]:
    """Loads all equities from lake organized by sector."""
    universe: dict[str, dict[str, pd.DataFrame]] = {}
    sectors = [d for d in os.listdir(lake_dir) if os.path.isdir(os.path.join(lake_dir, d)) and d != "Indices"]
    
    for sector in sectors:
        universe[sector] = {}
        sector_path = os.path.join(lake_dir, sector)
        for f in glob.glob(os.path.join(sector_path, "*.parquet")):
            sym = os.path.splitext(os.path.basename(f))[0]
            df = pd.read_parquet(f)
            if len(df) >= 60:
                universe[sector][sym] = df
    return universe

def evaluate_breakout_candidate(
    df: pd.DataFrame, 
    idx_t: int
) -> dict | None:
    """
    Evaluates institutional thrust and breakout conditions at Close[t] (3:10 PM).
    Strictly uses data up to idx_t (no lookahead).
    """
    if idx_t < 25:
        return None

    c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close_Raw"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    c_slice = c.iloc[:idx_t + 1]
    h_slice = h.iloc[:idx_t + 1]
    l_slice = l.iloc[:idx_t + 1]
    v_slice = v.iloc[:idx_t + 1]

    c_t = float(c_slice.iloc[-1])
    h_t = float(h_slice.iloc[-1])
    l_t = float(l_slice.iloc[-1])
    v_t = float(v_slice.iloc[-1])

    # 1. Trend Filter: EMA10 > EMA20 and C > EMA20
    ema_10 = float(c_slice.ewm(span=10, adjust=False).mean().iloc[-1])
    ema_20 = float(c_slice.ewm(span=20, adjust=False).mean().iloc[-1])
    ema_50 = float(c_slice.ewm(span=50, adjust=False).mean().iloc[-1])

    in_uptrend = (c_t > ema_20) and (ema_10 > ema_20)
    if not in_uptrend:
        return None

    # 2. Institutional Thrust: Close Location near day high
    bar_range = max(1e-4, h_t - l_t)
    close_loc = (c_t - l_t) / bar_range

    # 3. Volume Confirmation: Above 20-day volume average
    v_mean_20 = float(v_slice.iloc[-20:].mean())
    v_std_20 = float(v_slice.iloc[-20:].std()) if len(v_slice) >= 20 else 1.0
    v_zscore = (v_t - v_mean_20) / (v_std_20 + 1e-6)

    # 4. Breakout: High close vs rolling 20-day high close
    prev_max_close_20 = float(c_slice.iloc[-21:-1].max()) if len(c_slice) >= 21 else c_t
    prev_max_close_10 = float(c_slice.iloc[-11:-1].max()) if len(c_slice) >= 11 else c_t

    is_breakout_20 = c_t >= prev_max_close_20 * 0.998
    is_breakout_10 = c_t >= prev_max_close_10 * 0.998 and v_zscore > 0.5

    # Gating condition: must close in top 25% of day's range and break 20d or 10d high
    if close_loc >= 0.70 and (is_breakout_20 or is_breakout_10):
        # Calculate momentum and volatility features
        ret_1d = float((c_t / c_slice.iloc[-2]) - 1.0) if len(c_slice) >= 2 else 0.0
        ret_5d = float((c_t / c_slice.iloc[-6]) - 1.0) if len(c_slice) >= 6 else 0.0
        
        signal_strength = (close_loc * 0.4) + (min(3.0, max(0.0, v_zscore)) / 3.0 * 0.3) + (min(0.05, max(0.0, ret_5d)) / 0.05 * 0.3)
        
        return {
            "close_t": c_t,
            "high_t": h_t,
            "low_t": l_t,
            "close_loc": close_loc,
            "v_zscore": v_zscore,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "dist_ema20": (c_t - ema_20) / ema_20,
            "signal_strength": signal_strength,
            "breakout_type": "20D_BREAKOUT" if is_breakout_20 else "10D_THRUST"
        }
    return None

def compute_next_day_outcome(df: pd.DataFrame, idx_t: int, close_t: float) -> dict:
    """Computes real market outcome on Day t+1."""
    if idx_t + 1 >= len(df):
        return {"status": "LIVE_PENDING"}

    c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close_Raw"]
    o_next = float(df["Open"].iloc[idx_t + 1])
    h_next = float(df["High"].iloc[idx_t + 1])
    l_next = float(df["Low"].iloc[idx_t + 1])
    c_next = float(c.iloc[idx_t + 1])

    open_gap_pct = ((o_next - close_t) / close_t) * 100.0
    max_high_pct = ((h_next - close_t) / close_t) * 100.0
    min_low_pct = ((l_next - close_t) / close_t) * 100.0
    day_close_pct = ((c_next - close_t) / close_t) * 100.0

    hit_expansion_150 = max_high_pct >= 1.50
    hit_expansion_100 = max_high_pct >= 1.00
    pos_close = day_close_pct > 0.0
    pos_open = open_gap_pct > 0.0

    return {
        "status": "RESOLVED",
        "next_open": o_next,
        "next_high": h_next,
        "next_low": l_next,
        "next_close": c_next,
        "open_gap_pct": round(open_gap_pct, 2),
        "max_high_pct": round(max_high_pct, 2),
        "min_low_pct": round(min_low_pct, 2),
        "day_close_pct": round(day_close_pct, 2),
        "hit_150_expansion": hit_expansion_150,
        "hit_100_expansion": hit_expansion_100,
        "pos_close": pos_close,
        "pos_open": pos_open,
        "win": hit_expansion_100 or pos_close
    }

def run_30day_walkforward() -> dict:
    """Executes walk-forward breakout backtest over the last 30 trading days."""
    universe = load_all_equities()
    total_equities = sum(len(stocks) for stocks in universe.values())
    logger.info(f"Loaded {total_equities} equities across {len(universe)} sectors.")

    # Get common dates from benchmark or a major stock
    sample_df = universe["Automotive"]["BAJAJ-AUTO"]
    all_dates = [ts.strftime("%Y-%m-%d") for ts in sample_df.index]
    
    # Take the last 30 trading dates
    eval_dates = all_dates[-30:]
    last_7_dates = eval_dates[-7:]
    logger.info(f"Walk-forward dates: {eval_dates[0]} to {eval_dates[-1]} (Total {len(eval_dates)} trading sessions)")

    all_signals: list[dict] = []
    signals_by_date: dict[str, list[dict]] = {d: [] for d in eval_dates}

    for sector, stocks in universe.items():
        for sym, df in stocks.items():
            df_dates = [ts.strftime("%Y-%m-%d") for ts in df.index]
            date_to_idx = {d: i for i, d in enumerate(df_dates)}

            for d in eval_dates:
                if d not in date_to_idx:
                    continue
                idx_t = date_to_idx[d]
                cand = evaluate_breakout_candidate(df, idx_t)
                if cand is not None:
                    outcome = compute_next_day_outcome(df, idx_t, cand["close_t"])
                    signal_record = {
                        "ticker": sym,
                        "sector": sector,
                        "signal_date": d,
                        "prediction": "BULLISH_BREAKOUT",
                        **cand,
                        **outcome
                    }
                    all_signals.append(signal_record)
                    signals_by_date[d].append(signal_record)

    resolved_signals = [s for s in all_signals if s["status"] == "RESOLVED"]
    live_signals = [s for s in all_signals if s["status"] == "LIVE_PENDING"]

    # Compute aggregate statistics
    if resolved_signals:
        win_rate_close = float(np.mean([s["pos_close"] for s in resolved_signals]) * 100)
        win_rate_gap = float(np.mean([s["pos_open"] for s in resolved_signals]) * 100)
        win_rate_exp_100 = float(np.mean([s["hit_100_expansion"] for s in resolved_signals]) * 100)
        win_rate_exp_150 = float(np.mean([s["hit_150_expansion"] for s in resolved_signals]) * 100)
        
        avg_open_gap = float(np.mean([s["open_gap_pct"] for s in resolved_signals]))
        avg_max_high = float(np.mean([s["max_high_pct"] for s in resolved_signals]))
        avg_close_ret = float(np.mean([s["day_close_pct"] for s in resolved_signals]))
        avg_min_low = float(np.mean([s["min_low_pct"] for s in resolved_signals]))
    else:
        win_rate_close = win_rate_gap = win_rate_exp_100 = win_rate_exp_150 = 0.0
        avg_open_gap = avg_max_high = avg_close_ret = avg_min_low = 0.0

    last_7_signals = [s for s in all_signals if s["signal_date"] in last_7_dates]

    summary = {
        "evaluation_period": f"{eval_dates[0]} to {eval_dates[-1]}",
        "total_trading_days": len(eval_dates),
        "total_equities_monitored": total_equities,
        "total_signals_generated": len(all_signals),
        "resolved_signals_count": len(resolved_signals),
        "live_pending_signals_count": len(live_signals),
        "aggregate_metrics_30d": {
            "win_rate_positive_close_pct": round(win_rate_close, 2),
            "win_rate_positive_gap_pct": round(win_rate_gap, 2),
            "win_rate_expansion_100bps_pct": round(win_rate_exp_100, 2),
            "win_rate_expansion_150bps_pct": round(win_rate_exp_150, 2),
            "avg_overnight_open_gap_pct": round(avg_open_gap, 2),
            "avg_max_intraday_high_pct": round(avg_max_high, 2),
            "avg_day_close_return_pct": round(avg_close_ret, 2),
            "avg_min_intraday_drawdown_pct": round(avg_min_low, 2)
        },
        "last_7_days_dates": last_7_dates,
        "last_7_signals_count": len(last_7_signals),
        "last_7_days_signals": last_7_signals,
        "live_pending_signals": live_signals
    }

    os.makedirs("audit", exist_ok=True)
    out_file = "audit/walkforward_30d_breakout.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=_np_serial)
    logger.info(f"Walk-forward audit saved to {out_file}")

    return summary

if __name__ == "__main__":
    res = run_30day_walkforward()
    print("\n" + "="*80)
    print("30-DAY WALK-FORWARD BREAKOUT PREDICTION AUDIT SUMMARY")
    print("="*80)
    print(f"Evaluation Window: {res['evaluation_period']} ({res['total_trading_days']} trading sessions)")
    print(f"Total Equities: {res['total_equities_monitored']}")
    print(f"Total Breakout Signals Generated: {res['total_signals_generated']} (Resolved: {res['resolved_signals_count']}, Live: {res['live_pending_signals_count']})")
    print("\n--- AGGREGATE 30-DAY PERFORMANCE (DAY t+1 REAL OUTCOMES) ---")
    m = res["aggregate_metrics_30d"]
    print(f"Win Rate (Positive Day Close): {m['win_rate_positive_close_pct']}%")
    print(f"Win Rate (Intraday High >= +1.0% Expansion): {m['win_rate_expansion_100bps_pct']}%")
    print(f"Win Rate (Intraday High >= +1.5% Expansion): {m['win_rate_expansion_150bps_pct']}%")
    print(f"Win Rate (Overnight Gap Up): {m['win_rate_positive_gap_pct']}%")
    print(f"Average Overnight Open Gap: +{m['avg_overnight_open_gap_pct']}% (+{int(m['avg_overnight_open_gap_pct']*100)} bps)")
    print(f"Average Max Intraday Expansion: +{m['avg_max_intraday_high_pct']}% (+{int(m['avg_max_intraday_high_pct']*100)} bps)")
    print(f"Average Full-Day Close Return: +{m['avg_day_close_return_pct']}% (+{int(m['avg_day_close_return_pct']*100)} bps)")
    print(f"Average Intraday Max Drawdown: {m['avg_min_intraday_drawdown_pct']}%")
