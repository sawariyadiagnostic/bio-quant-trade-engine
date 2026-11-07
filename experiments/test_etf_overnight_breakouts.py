"""
ETF Overnight & Breakout Empirical Walk-Forward Engine.
Evaluates NIFTYBEES and BANKBEES over:
1. Last 30 Trading Days (Day-by-Day Granular Evaluation)
2. Last 6 Months (126 Trading Sessions, 2026-04-29 to 2026-10-29)
Strict zero-lookahead walk-forward test comparing:
- Overnight Drift (Close[t] -> Open[t+1])
- Intraday Expansion (Close[t] -> High[t+1])
- Full-Day Return (Close[t] -> Close[t+1])
- Breakout Signals vs Regime-Filtered vs Unconditional Holding
- Exact ETF Statutory Friction (0.012% / 1.2 bps round-trip)
"""
import os
import json
import numpy as np
import pandas as pd
from loguru import logger

ETF_ROUNDTRIP_FRICTION = 0.00012  # 1.2 bps (STT 0.001% sell + Stamp 0.005% buy + NSE fee 0.003% + GST)

def _np_serial(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    return str(obj)

def load_etf_data(symbol: str) -> pd.DataFrame:
    path = f"data/lake/Indices/{symbol}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found at {path}")
    df = pd.read_parquet(path)
    return df

def analyze_etf_overnight_and_breakouts(symbol: str, window_bars: int = 126) -> dict:
    """
    Analyzes overnight drift and breakout signals over the given window.
    window_bars=126 corresponds to 6 calendar months.
    """
    df = load_etf_data(symbol)
    if len(df) < window_bars + 30:
        raise ValueError(f"Insufficient data for {symbol}: {len(df)} bars")

    # Working slice
    work_df = df.iloc[-(window_bars + 30):].copy()
    c = work_df["Close_Adj"] if "Close_Adj" in work_df.columns else work_df["Close"]
    o = work_df["Open"]
    h = work_df["High"]
    l = work_df["Low"]
    v = work_df["Volume"]

    # Technical Indicators
    ema_10 = c.ewm(span=10, adjust=False).mean()
    ema_20 = c.ewm(span=20, adjust=False).mean()
    v_mean_20 = v.rolling(20).mean()
    v_std_20 = v.rolling(20).std().replace(0, np.nan)
    v_zscore = ((v - v_mean_20) / v_std_20).fillna(0.0)

    # Focus on the evaluation window
    eval_slice_indices = range(len(work_df) - window_bars, len(work_df))
    daily_records: list[dict] = []
    breakout_records: list[dict] = []

    for i in eval_slice_indices:
        dt = work_df.index[i].strftime("%Y-%m-%d")
        c_t = float(c.iloc[i])
        h_t = float(h.iloc[i])
        l_t = float(l.iloc[i])
        v_t = float(v.iloc[i])
        ema20_t = float(ema_20.iloc[i])
        ema10_t = float(ema_10.iloc[i])
        vz_t = float(v_zscore.iloc[i])

        in_uptrend = (c_t > ema20_t) and (ema10_t > ema20_t)
        bar_range = max(1e-4, h_t - l_t)
        close_loc = (c_t - l_t) / bar_range

        # 20-day high close check
        prev_20_max = float(c.iloc[i-20:i].max())
        is_breakout = (c_t >= prev_20_max * 0.999) and in_uptrend and (close_loc >= 0.70)

        # Check if t+1 exists
        if i + 1 < len(work_df):
            next_dt = work_df.index[i + 1].strftime("%Y-%m-%d")
            o_next = float(o.iloc[i + 1])
            h_next = float(h.iloc[i + 1])
            l_next = float(l.iloc[i + 1])
            c_next = float(c.iloc[i + 1])

            open_gap_pct = ((o_next - c_t) / c_t) * 100.0
            max_high_pct = ((h_next - c_t) / c_t) * 100.0
            min_low_pct = ((l_next - c_t) / c_t) * 100.0
            day_close_pct = ((c_next - c_t) / c_t) * 100.0

            record = {
                "signal_date": dt,
                "next_date": next_dt,
                "close_t": round(c_t, 2),
                "in_uptrend": in_uptrend,
                "is_breakout": is_breakout,
                "close_loc": round(close_loc, 2),
                "v_zscore": round(vz_t, 2),
                "open_gap_pct": round(open_gap_pct, 2),
                "max_high_pct": round(max_high_pct, 2),
                "min_low_pct": round(min_low_pct, 2),
                "day_close_pct": round(day_close_pct, 2),
                "pos_gap": open_gap_pct > 0.0,
                "pos_close": day_close_pct > 0.0,
                "status": "RESOLVED"
            }
        else:
            record = {
                "signal_date": dt,
                "next_date": "NEXT_SESSION",
                "close_t": round(c_t, 2),
                "in_uptrend": in_uptrend,
                "is_breakout": is_breakout,
                "close_loc": round(close_loc, 2),
                "v_zscore": round(vz_t, 2),
                "status": "LIVE_PENDING"
            }

        daily_records.append(record)
        if is_breakout:
            breakout_records.append(record)

    # Resolved sessions
    resolved_days = [r for r in daily_records if r["status"] == "RESOLVED"]
    resolved_breakouts = [r for r in breakout_records if r["status"] == "RESOLVED"]

    # Overnight statistics
    all_gaps = [r["open_gap_pct"] for r in resolved_days]
    all_closes = [r["day_close_pct"] for r in resolved_days]
    all_highs = [r["max_high_pct"] for r in resolved_days]

    uptrend_days = [r for r in resolved_days if r["in_uptrend"]]
    uptrend_gaps = [r["open_gap_pct"] for r in uptrend_days]

    # Compounding curve for unconditional overnight holding
    # Net return per day = (1 + gap) * (1 - friction) - 1
    on_returns = np.array(all_gaps) / 100.0
    cum_gross_on = float(np.prod(1.0 + on_returns) - 1.0) * 100.0
    cum_net_on = float(np.prod((1.0 + on_returns) * (1.0 - ETF_ROUNDTRIP_FRICTION)) - 1.0) * 100.0

    # Compounding curve for regime-filtered overnight (only in uptrend)
    regime_on_ret = np.array([r / 100.0 if d["in_uptrend"] else 0.0 for r, d in zip(all_gaps, resolved_days)])
    regime_friction = np.array([ETF_ROUNDTRIP_FRICTION if d["in_uptrend"] else 0.0 for d in resolved_days])
    cum_gross_regime_on = float(np.prod(1.0 + regime_on_ret) - 1.0) * 100.0
    cum_net_regime_on = float(np.prod((1.0 + regime_on_ret) * (1.0 - regime_friction)) - 1.0) * 100.0

    # Buy & Hold return over the period
    bnh_return = float((c.iloc[-1] / c.iloc[-window_bars]) - 1.0) * 100.0

    # Breakout specific metrics
    if resolved_breakouts:
        bo_gaps = [r["open_gap_pct"] for r in resolved_breakouts]
        bo_highs = [r["max_high_pct"] for r in resolved_breakouts]
        bo_closes = [r["day_close_pct"] for r in resolved_breakouts]
        bo_win_rate_exp80 = float(np.mean([h >= 0.80 for h in bo_highs]) * 100)
        bo_win_rate_close = float(np.mean([c_ret > 0.0 for c_ret in bo_closes]) * 100)
        bo_win_rate_gap = float(np.mean([g > 0.0 for g in bo_gaps]) * 100)
        avg_bo_gap = float(np.mean(bo_gaps))
        avg_bo_high = float(np.mean(bo_highs))
        avg_bo_close = float(np.mean(bo_closes))
    else:
        bo_win_rate_exp80 = bo_win_rate_close = bo_win_rate_gap = 0.0
        avg_bo_gap = avg_bo_high = avg_bo_close = 0.0

    # 30-day subset
    last_30_records = daily_records[-30:]
    last_30_resolved = [r for r in last_30_records if r["status"] == "RESOLVED"]
    last_30_gaps = [r["open_gap_pct"] for r in last_30_resolved]
    last_30_highs = [r["max_high_pct"] for r in last_30_resolved]
    last_30_closes = [r["day_close_pct"] for r in last_30_resolved]

    return {
        "symbol": symbol,
        "window_bars": window_bars,
        "period_start": work_df.index[-window_bars].strftime("%Y-%m-%d"),
        "period_end": work_df.index[-1].strftime("%Y-%m-%d"),
        "buy_and_hold_return_pct": round(bnh_return, 2),
        "overall_6m_metrics": {
            "total_sessions": len(resolved_days),
            "unconditional_overnight": {
                "gap_up_win_rate_pct": round(float(np.mean([g > 0.0 for g in all_gaps]) * 100), 2),
                "avg_open_gap_pct": round(float(np.mean(all_gaps)), 2),
                "avg_open_gap_bps": round(float(np.mean(all_gaps)) * 100, 1),
                "cumulative_gross_return_pct": round(cum_gross_on, 2),
                "cumulative_net_return_pct": round(cum_net_on, 2)
            },
            "regime_filtered_overnight": {
                "uptrend_sessions": len(uptrend_days),
                "gap_up_win_rate_pct": round(float(np.mean([g > 0.0 for g in uptrend_gaps]) * 100), 2) if uptrend_gaps else 0.0,
                "avg_open_gap_pct": round(float(np.mean(uptrend_gaps)), 2) if uptrend_gaps else 0.0,
                "avg_open_gap_bps": round(float(np.mean(uptrend_gaps)) * 100, 1) if uptrend_gaps else 0.0,
                "cumulative_gross_return_pct": round(cum_gross_regime_on, 2),
                "cumulative_net_return_pct": round(cum_net_regime_on, 2)
            },
            "breakout_signals": {
                "total_breakouts": len(breakout_records),
                "resolved_count": len(resolved_breakouts),
                "win_rate_expansion_80bps_pct": round(bo_win_rate_exp80, 2),
                "win_rate_positive_close_pct": round(bo_win_rate_close, 2),
                "win_rate_gap_up_pct": round(bo_win_rate_gap, 2),
                "avg_open_gap_pct": round(avg_bo_gap, 2),
                "avg_max_high_pct": round(avg_bo_high, 2),
                "avg_day_close_pct": round(avg_bo_close, 2)
            }
        },
        "last_30_days_metrics": {
            "total_sessions": len(last_30_resolved),
            "gap_up_win_rate_pct": round(float(np.mean([g > 0.0 for g in last_30_gaps]) * 100), 2),
            "avg_open_gap_pct": round(float(np.mean(last_30_gaps)), 2),
            "avg_open_gap_bps": round(float(np.mean(last_30_gaps)) * 100, 1),
            "avg_max_high_pct": round(float(np.mean(last_30_highs)), 2),
            "avg_day_close_pct": round(float(np.mean(last_30_closes)), 2),
            "breakouts_in_30d": [r for r in last_30_records if r.get("is_breakout")]
        },
        "last_30_days_daily_log": last_30_records
    }

def run_etf_experiments():
    results = {}
    for sym in ["NIFTYBEES", "BANKBEES"]:
        logger.info(f"Analyzing {sym} for last 30 days & 6 months...")
        res = analyze_etf_overnight_and_breakouts(sym, window_bars=126)
        results[sym] = res

    os.makedirs("audit", exist_ok=True)
    out_file = "audit/etf_overnight_breakout_audit.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=_np_serial)
    logger.info(f"Saved ETF audit to {out_file}")
    return results

if __name__ == "__main__":
    res = run_etf_experiments()
    for sym, d in res.items():
        print("\n" + "="*80)
        print(f"ETF EMPIRICAL REPORT: {sym} ({d['period_start']} to {d['period_end']})")
        print("="*80)
        print(f"Buy & Hold 6-Month Return: {d['buy_and_hold_return_pct']}%")
        
        m6 = d["overall_6m_metrics"]
        print("\n--- 6-MONTH OVERNIGHT DRIFT PERFORMANCE ---")
        u = m6["unconditional_overnight"]
        print(f"Unconditional Overnight (Every Day): Win Rate: {u['gap_up_win_rate_pct']}% | Avg Gap: {u['avg_open_gap_pct']}% ({u['avg_open_gap_bps']} bps)")
        print(f"  -> Cumulative Gross: {u['cumulative_gross_return_pct']}% | Net (with 1.2 bps ETF STT): {u['cumulative_net_return_pct']}%")
        
        r = m6["regime_filtered_overnight"]
        print(f"Regime-Filtered Overnight (Only in Trend): Uptrend Days: {r['uptrend_sessions']}/{m6['total_sessions']} | Win Rate: {r['gap_up_win_rate_pct']}% | Avg Gap: {r['avg_open_gap_pct']}% ({r['avg_open_gap_bps']} bps)")
        print(f"  -> Cumulative Gross: {r['cumulative_gross_return_pct']}% | Net (with 1.2 bps ETF STT): {r['cumulative_net_return_pct']}%")

        b = m6["breakout_signals"]
        print("\n--- 6-MONTH BREAKOUT SIGNALS ---")
        print(f"Total Breakouts Triggered: {b['total_breakouts']} (Resolved: {b['resolved_count']})")
        print(f"  -> Next-Day Win Rate (Exp >= +0.80% / 200 pts): {b['win_rate_expansion_80bps_pct']}%")
        print(f"  -> Next-Day Win Rate (Close > Entry): {b['win_rate_positive_close_pct']}%")
        print(f"  -> Next-Day Win Rate (Gap Up): {b['win_rate_gap_up_pct']}%")
        print(f"  -> Avg Open Gap: {b['avg_open_gap_pct']}% | Avg Max High: {b['avg_max_high_pct']}% | Avg Close Ret: {b['avg_day_close_pct']}%")

        m30 = d["last_30_days_metrics"]
        print("\n--- LAST 30 DAYS PERFORMANCE ---")
        print(f"Sessions: {m30['total_sessions']} | Gap Up Win Rate: {m30['gap_up_win_rate_pct']}% | Avg Gap: {m30['avg_open_gap_pct']}% ({m30['avg_open_gap_bps']} bps)")
        print(f"Avg Max High: {m30['avg_max_high_pct']}% | Avg Close Ret: {m30['avg_day_close_pct']}%")
        print(f"Breakouts in last 30d: {len(m30['breakouts_in_30d'])}")
