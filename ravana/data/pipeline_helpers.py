"""Helper functions and metadata builders for batch pipeline orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def sanitize_filename(ticker: str) -> str:
    """Normalize ticker string into a filesystem-safe file basename."""
    clean = ticker.replace("^", "").replace(".NS", "").replace("=X", "").replace("=F", "")
    if ticker.upper() == "INR=X":
        return "USDINR"
    return clean


def make_meta(ticker: str, df: pd.DataFrame, fpath: Path) -> Dict[str, Any]:
    """Construct audit metadata dictionary for a stored parquet file."""
    return {
        "ticker": ticker,
        "bars": len(df),
        "start": str(df.index[0]) if not df.empty else None,
        "end": str(df.index[-1]) if not df.empty else None,
        "file_size_bytes": fpath.stat().st_size if fpath.exists() else 0,
    }


def enrich_intraday_for_m7(
    base: str,
    df_clean: pd.DataFrame,
    m4_dir: Path,
) -> pd.DataFrame:
    """Merge tracking spread analytics and causally lagged macro series into 5m dataframe.
    
    Adheres strictly to Medallion architecture: macro drivers are sourced from m4_cleaned/daily_1d.
    """
    df = df_clean.copy()
    spread_path = m4_dir / "intraday_5m" / f"{base}_spread.parquet"
    if spread_path.exists():
        df_spread = pd.read_parquet(spread_path)
        for col in ["Theoretical_Close", "Spread_INR"]:
            if col in df_spread.columns:
                df[col] = df_spread[col]

    unique_dates = sorted(list(set(t.date() for t in df.index)))
    bar_dates = pd.Series([t.date() for t in df.index], index=df.index)
    macro_specs = [
        ("VIX", "VIX_clean.parquet", 20.0),
        ("USDINR", "USDINR_clean.parquet", 85.0),
        ("SP500", "GSPC_clean.parquet", 5000.0),
    ]
    for name, fname, default_v in macro_specs:
        m_path = m4_dir / "daily_1d" / fname
        if not m_path.exists():
            continue
        df_m = pd.read_parquet(m_path)
        s = pd.Series(df_m["Close"].values, index=[t.date() for t in df_m.index]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        r = np.log(s / s.shift(1)).fillna(0.0)
        lvl_map: Dict[Any, float] = {}
        ret_map: Dict[Any, float] = {}
        for d in unique_dates:
            prior_s = s[s.index < d]
            lvl_map[d] = float(prior_s.iloc[-1]) if len(prior_s) > 0 else default_v
            prior_r = r[r.index < d]
            ret_map[d] = float(prior_r.iloc[-1]) if len(prior_r) > 0 else 0.0
        df[f"Macro_{name}"] = bar_dates.map(lvl_map)
        df[f"Macro_{name}_RET"] = bar_dates.map(ret_map)
    return df
