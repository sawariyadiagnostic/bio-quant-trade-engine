import json
import numpy as np

with open('audit/full_universe_analysis.json') as f:
    data = json.load(f)

sector_stats = []
active_stocks = []
all_stocks = []

for sector, stocks in data.items():
    n_stocks = len(stocks)
    if n_stocks == 0:
        continue
    rets = [s['5y_cum_return'] for s in stocks.values()]
    vols = [s['ann_volatility'] for s in stocks.values()]
    mdds = [s['max_drawdown'] for s in stocks.values()]
    noises = [s['noise_ratio_pct'] for s in stocks.values()]
    active_cnt = sum(1 for s in stocks.values() if not s['is_noise'])
    noise_cnt = n_stocks - active_cnt
    
    sector_stats.append({
        'sector': sector,
        'stocks': n_stocks,
        'avg_ret': float(np.mean(rets)),
        'avg_vol': float(np.mean(vols)),
        'avg_mdd': float(np.mean(mdds)),
        'avg_noise': float(np.mean(noises)),
        'active': active_cnt,
        'noise': noise_cnt
    })
    
    for ticker, s in stocks.items():
        all_stocks.append((ticker, sector, s))
        if not s['is_noise']:
            active_stocks.append((ticker, sector, s))

print("=== SECTOR SUMMARY ===")
print(f"{'Sector':<26} {'Count':<6} {'Avg 5Y Ret':<12} {'Avg Vol':<10} {'Avg MDD':<10} {'Noise Ratio':<12} {'Active':<8} {'Cash/Noise':<10}")
print("-" * 95)
for s in sorted(sector_stats, key=lambda x: x['avg_ret'], reverse=True):
    print(f"{s['sector']:<26} {s['stocks']:<6} {s['avg_ret']*100:+10.1f}% {s['avg_vol']*100:8.1f}% {s['avg_mdd']*100:8.1f}% {s['avg_noise']:10.1f}% {s['active']:<8} {s['noise']:<10}")

print("\n=== TOP 20 ASSETS BY 5-YEAR PERFORMANCE ===")
print(f"{'Ticker':<14} {'Sector':<24} {'Regime':<8} {'5Y Return':<12} {'Vol':<10} {'MDD':<10} {'Noise %':<10}")
print("-" * 90)
for ticker, sector, s in sorted(all_stocks, key=lambda x: x[2]['5y_cum_return'], reverse=True)[:20]:
    reg_str = f"R{s['today_regime']}" if not s['is_noise'] else "NOISE(-1)"
    print(f"{ticker:<14} {sector:<24} {reg_str:<8} {s['5y_cum_return']*100:+10.1f}% {s['ann_volatility']*100:8.1f}% {s['max_drawdown']*100:8.1f}% {s['noise_ratio_pct']:8.1f}%")

print("\n=== CURRENTLY ACTIVE REGIME EQUITIES (REGIME >= 0) ===")
print(f"{'Ticker':<14} {'Sector':<24} {'Regime':<8} {'5Y Return':<12} {'Vol':<10} {'MDD':<10} {'Noise %':<10}")
print("-" * 90)
for ticker, sector, s in sorted(active_stocks, key=lambda x: x[2]['5y_cum_return'], reverse=True):
    reg_str = f"R{s['today_regime']}"
    print(f"{ticker:<14} {sector:<24} {reg_str:<8} {s['5y_cum_return']*100:+10.1f}% {s['ann_volatility']*100:8.1f}% {s['max_drawdown']*100:8.1f}% {s['noise_ratio_pct']:8.1f}%")

print(f"\nTotal Equities Analyzed: {len(all_stocks)}")
print(f"Total Active Regime Candidates: {len(active_stocks)}")
print(f"Total Noise (100% Cash Gated): {len(all_stocks) - len(active_stocks)}")
