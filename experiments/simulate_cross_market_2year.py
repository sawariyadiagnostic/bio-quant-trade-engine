"""
2-Year Cross-Market Lead-Lag & ADR Arbitrage Empirical Compounding Engine.
Simulates 2 full years (Oct 2024 to Oct 2026, ~480 aligned trading sessions) across:
1. S&P 500 -> NIFTYBEES
2. NASDAQ -> NIFTYBEES
3. INFY ADR (NYSE) -> INFY.NS (NSE)
4. HDFC ADR (NYSE: HDB) -> HDFCBANK.NS (NSE)
5. ICICI ADR (NYSE: IBN) -> ICICIBANK.NS (NSE)
Compares:
  - Baseline Buy & Hold
  - Unconditional Overnight (All days)
  - US Lead-Lag Filtered Overnight (Only enter when US / ADR is GREEN, cash when RED)
  - 1x, 2x, 3x, and 5x Leverage Tiers starting from INR 10,000
"""
import os
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

def load_aligned_pair(
    india_path: str, 
    us_path: str
) -> pd.DataFrame:
    """Aligns Indian Day t close & Day t+1 open with US Day t close."""
    df_ind = pd.read_parquet(india_path)
    df_us = pd.read_parquet(us_path)

    df_ind["dt"] = df_ind.index.strftime("%Y-%m-%d")
    df_us["dt"] = df_us.index.strftime("%Y-%m-%d")

    c_ind = df_ind["Close_Adj"] if "Close_Adj" in df_ind.columns else df_ind["Close"]
    o_ind = df_ind["Open"]
    c_us = df_us["Close_Adj"] if "Close_Adj" in df_us.columns else df_us["Close"]

    us_rets = (c_us / c_us.shift(1) - 1.0)
    us_map = dict(zip(df_us["dt"], us_rets))

    records = []
    for t in range(len(df_ind) - 1):
        d_today = df_ind["dt"].iloc[t]
        d_next = df_ind["dt"].iloc[t + 1]

        if d_today in us_map and np.isfinite(us_map[d_today]):
            c_today = float(c_ind.iloc[t])
            o_next = float(o_ind.iloc[t + 1])
            c_next = float(c_ind.iloc[t + 1])

            us_r = float(us_map[d_today])
            ind_gap = (o_next - c_today) / c_today
            ind_close = (c_next - c_today) / c_today

            records.append({
                "date_t": d_today,
                "date_next": d_next,
                "us_ret": us_r,
                "us_ret_pct": round(us_r * 100.0, 2),
                "ind_gap": ind_gap,
                "ind_gap_pct": round(ind_gap * 100.0, 2),
                "ind_close_pct": round(ind_close * 100.0, 2),
                "us_green": us_r > 0.0,
                "ind_gap_green": ind_gap > 0.0
            })

    return pd.DataFrame(records)

def evaluate_pair_statistics(df_aligned: pd.DataFrame, label: str) -> dict:
    """Computes 2-year statistical lead-lag predictability metrics."""
    total = len(df_aligned)
    us_rets = df_aligned["us_ret"].to_numpy()
    ind_gaps = df_aligned["ind_gap"].to_numpy()

    corr = float(np.corrcoef(us_rets, ind_gaps)[0, 1])
    match_rate = float(np.mean((us_rets > 0) == (ind_gaps > 0)) * 100.0)

    # When US is Green
    green_mask = df_aligned["us_green"]
    green_df = df_aligned[green_mask]
    green_win_rate = float(np.mean(green_df["ind_gap_green"]) * 100.0)
    avg_gap_green_bps = float(green_df["ind_gap"].mean()) * 10000.0

    # When US is Red
    red_df = df_aligned[~green_mask]
    red_gap_down_rate = float(np.mean(~red_df["ind_gap_green"]) * 100.0)
    avg_gap_red_bps = float(red_df["ind_gap"].mean()) * 10000.0

    # Strong US Signals (>= +0.5% and <= -0.5%)
    strong_green = df_aligned[df_aligned["us_ret"] >= 0.005]
    strong_green_win = float(np.mean(strong_green["ind_gap_green"]) * 100.0) if len(strong_green) > 0 else 0.0
    strong_green_gap_bps = float(strong_green["ind_gap"].mean() * 10000.0) if len(strong_green) > 0 else 0.0

    strong_red = df_aligned[df_aligned["us_ret"] <= -0.005]
    strong_red_down = float(np.mean(~strong_red["ind_gap_green"]) * 100.0) if len(strong_red) > 0 else 0.0
    strong_red_gap_bps = float(strong_red["ind_gap"].mean() * 10000.0) if len(strong_red) > 0 else 0.0

    return {
        "pair": label,
        "total_sessions": total,
        "pearson_corr": round(corr, 3),
        "directional_match_pct": round(match_rate, 2),
        "us_green_stats": {
            "sessions": len(green_df),
            "india_gap_up_pct": round(green_win_rate, 2),
            "avg_gap_bps": round(avg_gap_green_bps, 1)
        },
        "us_red_stats": {
            "sessions": len(red_df),
            "india_gap_down_pct": round(red_gap_down_rate, 2),
            "avg_gap_bps": round(avg_gap_red_bps, 1)
        },
        "strong_us_green_stats": {
            "sessions": len(strong_green),
            "india_gap_up_pct": round(strong_green_win, 2),
            "avg_gap_bps": round(strong_green_gap_bps, 1)
        },
        "strong_us_red_stats": {
            "sessions": len(strong_red),
            "india_gap_down_pct": round(strong_red_down, 2),
            "avg_gap_bps": round(strong_red_gap_bps, 1)
        }
    }

def simulate_strategy_compounding(
    df_aligned: pd.DataFrame, 
    initial_capital: float = 10000.0,
    friction: float = 0.00012,
    leverage: float = 1.0,
    use_lead_lag_filter: bool = True
) -> dict:
    """
    Simulates compounding returns over 2 years:
    If use_lead_lag_filter is True: Only trade when US Day t return > 0.0; otherwise stay in 100% Cash.
    """
    capital = initial_capital
    curve = [capital]
    trades = []

    for _, row in df_aligned.iterrows():
        us_up = row["us_green"]
        gap = row["ind_gap"]

        if use_lead_lag_filter and not us_up:
            # Stood down in 100% Cash! Dodged potential gap down
            curve.append(capital)
            continue

        # Enter trade
        net_ret = (gap * leverage) - (friction * leverage)
        if leverage > 1.0:
            net_ret -= (leverage - 1.0) * (0.09 / 365.0)  # margin interest

        pnl = capital * net_ret
        capital = max(0.0, capital + pnl)
        curve.append(capital)

        trades.append({
            "date": row["date_next"],
            "us_ret_pct": row["us_ret_pct"],
            "ind_gap_pct": row["ind_gap_pct"],
            "pnl": round(pnl, 2),
            "balance": round(capital, 2),
            "win": gap > 0.0
        })

    eq_arr = np.array(curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peaks) / (peaks + 1e-6)
    mdd = float(np.min(drawdowns)) * 100.0

    wins = sum(1 for tr in trades if tr["win"])
    win_rate = (wins / len(trades)) * 100.0 if trades else 0.0

    return {
        "filtered": use_lead_lag_filter,
        "leverage": leverage,
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "net_profit": round(capital - initial_capital, 2),
        "return_pct": round(((capital - initial_capital) / initial_capital) * 100.0, 2),
        "total_trades": len(trades),
        "winning_trades": wins,
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(mdd, 2)
    }

def run_2year_lead_lag_suite():
    pairs = [
        {
            "label": "S&P 500 -> NIFTYBEES",
            "india_path": "data/lake/Indices/NIFTYBEES.parquet",
            "us_path": "data/lake/Global/SP500.parquet",
            "friction": 0.00012
        },
        {
            "label": "NASDAQ -> NIFTYBEES",
            "india_path": "data/lake/Indices/NIFTYBEES.parquet",
            "us_path": "data/lake/Global/NASDAQ.parquet",
            "friction": 0.00012
        },
        {
            "label": "INFY ADR (NYSE) -> INFY.NS (NSE)",
            "india_path": "data/lake/Information_Technology/INFY.parquet",
            "us_path": "data/lake/Global/INFY_ADR.parquet",
            "friction": 0.00025
        },
        {
            "label": "HDFC ADR (NYSE) -> HDFCBANK.NS (NSE)",
            "india_path": "data/lake/Banking_Financials/HDFCBANK.parquet",
            "us_path": "data/lake/Global/HDFC_ADR.parquet",
            "friction": 0.00025
        },
        {
            "label": "ICICI ADR (NYSE) -> ICICIBANK.NS (NSE)",
            "india_path": "data/lake/Banking_Financials/ICICIBANK.parquet",
            "us_path": "data/lake/Global/ICICI_ADR.parquet",
            "friction": 0.00025
        }
    ]

    all_results = {}

    for p in pairs:
        logger.info(f"Processing 2-year alignment for {p['label']}...")
        df_al = load_aligned_pair(p["india_path"], p["us_path"])
        stats = evaluate_pair_statistics(df_al, p["label"])

        # Run compounding simulations across leverage tiers (1x, 2x, 3x, 5x)
        # Compare Unfiltered vs US-Filtered
        simulations = {}
        for lev in [1.0, 2.0, 3.0, 5.0]:
            unfiltered = simulate_strategy_compounding(
                df_al, initial_capital=10000.0, friction=p["friction"], leverage=lev, use_lead_lag_filter=False
            )
            filtered = simulate_strategy_compounding(
                df_al, initial_capital=10000.0, friction=p["friction"], leverage=lev, use_lead_lag_filter=True
            )
            simulations[f"{lev:.0f}x"] = {
                "unfiltered": unfiltered,
                "us_lead_lag_filtered": filtered
            }

        all_results[p["label"]] = {
            "statistics": stats,
            "simulations": simulations
        }

    os.makedirs("audit", exist_ok=True)
    out_file = "audit/cross_market_2year_audit.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=_np_serial)
    logger.info(f"Saved 2-year cross-market audit to {out_file}")
    return all_results

if __name__ == "__main__":
    res = run_2year_lead_lag_suite()
    
    print("\n" + "="*95)
    print("2-YEAR CROSS-MARKET LEAD-LAG EMPIRICAL PREDICTABILITY REPORT (~480 SESSIONS)")
    print("="*95)
    for pair_name, d in res.items():
        st = d["statistics"]
        print(f"\n>>> PAIR: {pair_name}")
        print(f"    Sessions: {st['total_sessions']} | Pearson Corr: {st['pearson_corr']} | Directional Match: {st['directional_match_pct']}%")
        ug = st["us_green_stats"]
        ur = st["us_red_stats"]
        print(f"    When US is GREEN: India Gap Up Win Rate: {ug['india_gap_up_pct']}% | Avg Gap: +{ug['avg_gap_bps']} bps ({ug['sessions']} sessions)")
        print(f"    When US is RED:   India Gap Down Rate:   {ur['india_gap_down_pct']}% | Avg Gap: {ur['avg_gap_bps']} bps ({ur['sessions']} sessions)")
        
        sg = st["strong_us_green_stats"]
        sr = st["strong_us_red_stats"]
        print(f"    US RALLY (>= +0.5%): India Gap Up Win Rate: {sg['india_gap_up_pct']}% | Avg Gap: +{sg['avg_gap_bps']} bps")
        print(f"    US DUMP  (<= -0.5%): India Gap Down Rate:   {sr['india_gap_down_pct']}% | Avg Gap: {sr['avg_gap_bps']} bps")

        print("    " + "-"*85)
        print(f"    {'Leverage':8} | {'Unfiltered Final':18} | {'Unfilt MDD':10} | {'US-Filtered Final':20} | {'Filtered MDD':12} | {'Filtered Win Rate':16}")
        print("    " + "-"*85)
        for lev_key, sim in d["simulations"].items():
            u = sim["unfiltered"]
            f = sim["us_lead_lag_filtered"]
            print(f"    {lev_key:8} | INR {u['final_capital']:11,.2f} ({u['return_pct']:+6.1f}%) | {u['max_drawdown_pct']:7.2f}% | INR {f['final_capital']:11,.2f} ({f['return_pct']:+6.1f}%) | {f['max_drawdown_pct']:9.2f}% | {f['win_rate_pct']:6.2f}% ({f['winning_trades']}/{f['total_trades']})")
