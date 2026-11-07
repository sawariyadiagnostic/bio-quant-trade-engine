"""
1-Year Daily Overnight Simulation with Dynamic Trailing Stop & Maximum Leverage.
Evaluates 1 full year (250 trading sessions: Oct 2025 - Oct 2026) across:
1. NIFTYBEES (NSE Nifty 50 ETF)
2. KO (The Coca-Cola Company, NYSE)
3. VBL.NS (Varun Beverages, Indian Beverage Titan)
Compares:
- Baseline Buy & Hold
- Pure 9:15 AM Open Autoseller (1x, 2x, 3x, 5x leverage)
- Dynamic Morning Trailing Stop-Loss (ratchets up with price, autocloses on reversal)
- Maximum Safe Leverage Analysis (based on worst historical overnight gap down)
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

def load_asset_1year(parquet_path: str, bars: int = 250) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if len(df) < bars:
        raise ValueError(f"Insufficient bars in {parquet_path}: {len(df)} < {bars}")
    return df.iloc[-bars:].copy()

def simulate_overnight_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    leverage: float = 1.0,
    friction_pct: float = 0.00012,
    mode: str = "PURE_OPEN",
    trail_pct: float = 0.002,       # 0.20% trailing buffer
    stop_loss_pct: float = 0.0025   # 0.25% hard initial stop
) -> dict:
    """
    Executes 1-year daily overnight strategy with optional trailing stop and leverage.
    mode: 'PURE_OPEN', 'TRAILING_STOP', or 'TIGHT_CUT_OPEN'
    """
    c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]

    capital = initial_capital
    trade_logs: list[dict] = []
    equity_curve: list[float] = [capital]

    for t in range(len(df) - 1):
        dt_buy = df.index[t].strftime("%Y-%m-%d")
        dt_sell = df.index[t+1].strftime("%Y-%m-%d")
        c_buy = float(c.iloc[t])
        o_sell = float(o.iloc[t+1])
        h_sell = float(h.iloc[t+1])
        l_sell = float(l.iloc[t+1])
        c_sell = float(c.iloc[t+1])

        gap = (o_sell - c_buy) / c_buy
        max_h = (h_sell - c_buy) / c_buy
        min_l = (l_sell - c_buy) / c_buy

        if mode == "PURE_OPEN":
            # Sell 100% at Open unconditionally
            ret = gap
            exit_type = "OPEN_AUCTION"
        elif mode == "TIGHT_CUT_OPEN":
            # If gap down, exit at Open; if gap up, exit at Open
            ret = gap
            exit_type = "PROFIT_OPEN" if gap >= 0 else "LOSS_CUT_OPEN"
        elif mode == "TRAILING_STOP":
            # Dynamic Trailing Stop
            if gap <= -stop_loss_pct:
                # Gapped down below stop: filled at Open auction
                ret = gap
                exit_type = "GAP_STOP_OPEN"
            elif gap < 0:
                # Opened slightly down: if dips below stop, cut at -stop_loss_pct
                if min_l <= -stop_loss_pct:
                    ret = -stop_loss_pct
                    exit_type = "INTRADAY_STOP"
                else:
                    ret = (c_sell - c_buy) / c_buy
                    exit_type = "CLOSE_EXIT"
            else:
                # Gapped up! Ratchet trailing stop upward with high
                peak_ret = max_h
                trailing_trigger = peak_ret - trail_pct
                
                # Check if price retraced by trail_pct from high
                bar_retrace = (h_sell - l_sell) / c_buy
                if bar_retrace >= trail_pct and trailing_trigger >= gap:
                    # Trailed and locked profit
                    ret = trailing_trigger
                    exit_type = "TRAILING_STOP_LOCKED"
                elif bar_retrace >= trail_pct and trailing_trigger < gap:
                    # Trailing hit near open
                    ret = max(0.0, gap)
                    exit_type = "OPEN_PROTECT_EXIT"
                else:
                    # Maintained high till close
                    ret = (c_sell - c_buy) / c_buy
                    exit_type = "TREND_RUN_CLOSE"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply leverage and transaction friction
        # Total position size = capital * leverage
        net_ret = (ret * leverage) - (friction_pct * leverage)

        # Margin interest for leverage > 1.0 (approx 9% p.a. on borrowed capital overnight: 9% / 365)
        if leverage > 1.0:
            margin_interest = (leverage - 1.0) * (0.09 / 365.0)
            net_ret -= margin_interest

        pnl = capital * net_ret
        capital = max(0.0, capital + pnl)
        equity_curve.append(capital)

        trade_logs.append({
            "trade_num": t + 1,
            "buy_date": dt_buy,
            "sell_date": dt_sell,
            "buy_price": round(c_buy, 2),
            "exit_ret_pct": round(ret * 100.0, 2),
            "exit_type": exit_type,
            "pnl": round(pnl, 2),
            "balance": round(capital, 2),
            "win": ret > 0.0
        })

        if capital <= 10.0:  # Liquidation guard
            break

    wins = [tr for tr in trade_logs if tr["win"]]
    win_rate = (len(wins) / len(trade_logs)) * 100.0 if trade_logs else 0.0

    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peaks) / (peaks + 1e-6)
    mdd = float(np.min(drawdowns)) * 100.0

    # Max single gap down in history
    all_gaps = [(float(o.iloc[i+1]) - float(c.iloc[i])) / float(c.iloc[i]) for i in range(len(df)-1)]
    worst_gap_down = float(np.min(all_gaps)) * 100.0

    return {
        "mode": mode,
        "leverage": leverage,
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "net_profit": round(capital - initial_capital, 2),
        "return_pct": round(((capital - initial_capital) / initial_capital) * 100.0, 2),
        "win_rate_pct": round(win_rate, 2),
        "total_trades": len(trade_logs),
        "max_drawdown_pct": round(mdd, 2),
        "worst_overnight_gap_down_pct": round(worst_gap_down, 2),
        "max_safe_leverage": round(50.0 / abs(worst_gap_down), 1) if abs(worst_gap_down) > 0.01 else 10.0,
        "trade_logs": trade_logs[-15:]  # last 15 trades
    }

def run_asset_suite(
    asset_name: str, 
    parquet_path: str, 
    friction_pct: float
) -> dict:
    """Evaluates all modes and leverage tiers for a single asset over 1 year (250 bars)."""
    df = load_asset_1year(parquet_path, bars=250)
    
    # 1. Buy & Hold benchmark
    c = df["Close_Adj"] if "Close_Adj" in df.columns else df["Close"]
    bnh_ret = float((c.iloc[-1] / c.iloc[0]) - 1.0) * 100.0

    # 2. Pure Open Autoseller across leverage tiers (1x, 2x, 3x, 5x)
    leverage_results = {}
    for lev in [1.0, 2.0, 3.0, 4.0, 5.0]:
        res_lev = simulate_overnight_strategy(
            df, initial_capital=10000.0, leverage=lev, friction_pct=friction_pct, mode="PURE_OPEN"
        )
        leverage_results[f"{lev:.0f}x"] = res_lev

    # 3. Dynamic Trailing Stop-Loss (1x and 2x leverage)
    trail_1x = simulate_overnight_strategy(
        df, initial_capital=10000.0, leverage=1.0, friction_pct=friction_pct, 
        mode="TRAILING_STOP", trail_pct=0.002, stop_loss_pct=0.0025
    )
    trail_2x = simulate_overnight_strategy(
        df, initial_capital=10000.0, leverage=2.0, friction_pct=friction_pct, 
        mode="TRAILING_STOP", trail_pct=0.002, stop_loss_pct=0.0025
    )

    return {
        "asset": asset_name,
        "bars_evaluated": len(df),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
        "buy_and_hold_return_pct": round(bnh_ret, 2),
        "pure_open_by_leverage": leverage_results,
        "trailing_stop_1x": trail_1x,
        "trailing_stop_2x": trail_2x
    }

def run_all_1year_experiments():
    assets = [
        {"name": "NIFTYBEES", "path": "data/lake/Indices/NIFTYBEES.parquet", "friction": 0.00012},
        {"name": "KO_CocaCola", "path": "data/lake/Global/KO.parquet", "friction": 0.00005},
        {"name": "VBL_VarunBeverages", "path": "data/lake/Beverages/VBL.parquet", "friction": 0.00025}
    ]

    suite_results = {}
    for a in assets:
        logger.info(f"Running 1-year analysis for {a['name']}...")
        res = run_asset_suite(a["name"], a["path"], a["friction"])
        suite_results[a["name"]] = res

    os.makedirs("audit", exist_ok=True)
    out_file = "audit/trailing_stop_leverage_1year_audit.json"
    with open(out_file, "w") as f:
        json.dump(suite_results, f, indent=2, default=_np_serial)
    logger.info(f"Saved 1-year leverage audit to {out_file}")
    return suite_results

if __name__ == "__main__":
    res = run_all_1year_experiments()
    for name, d in res.items():
        print("\n" + "="*80)
        print(f"1-YEAR TRAILING STOP & LEVERAGE AUDIT: {name}")
        print(f"Window: {d['start_date']} to {d['end_date']} (249 completed daily overnight trades)")
        print(f"Buy & Hold Return: {d['buy_and_hold_return_pct']}%")
        print("="*80)
        
        print("\n--- PURE 9:15 AM AUTOSELLER BY LEVERAGE TIER ---")
        for lev_key, lres in d["pure_open_by_leverage"].items():
            print(f"[{lev_key} Leverage] Final: INR/USD {lres['final_capital']:,.2f} ({lres['return_pct']:+.2f}%) | Win Rate: {lres['win_rate_pct']}% | MDD: {lres['max_drawdown_pct']}%")

        tr1 = d["trailing_stop_1x"]
        tr2 = d["trailing_stop_2x"]
        print("\n--- DYNAMIC TRAILING STOP (0.2% TRAIL / 0.25% STOP) ---")
        print(f"[1x Trail] Final: INR/USD {tr1['final_capital']:,.2f} ({tr1['return_pct']:+.2f}%) | Win Rate: {tr1['win_rate_pct']}% | MDD: {tr1['max_drawdown_pct']}%")
        print(f"[2x Trail] Final: INR/USD {tr2['final_capital']:,.2f} ({tr2['return_pct']:+.2f}%) | Win Rate: {tr2['win_rate_pct']}% | MDD: {tr2['max_drawdown_pct']}%")
        print(f"Worst Single Overnight Gap Down: {tr1['worst_overnight_gap_down_pct']}%")
        print(f"Calculated Maximum Safe Leverage Ceiling: {tr1['max_safe_leverage']}x")
