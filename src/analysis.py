"""Analytical utilities for ABC classification and velocity metrics."""
from __future__ import annotations

import pandas as pd
import numpy as np


def compute_revenue(products: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    total_sales = daily.groupby("product_id", as_index=False)["sales"].sum()
    merged = products.merge(total_sales, on="product_id", how="left").fillna({"sales": 0})
    merged["annual_revenue"] = merged["sales"] * merged["unit_cost"]
    return merged


def abc_classification(df: pd.DataFrame, value_col: str = "annual_revenue") -> pd.DataFrame:
    ordered = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    total = ordered[value_col].sum() or 1.0
    ordered["cum_pct"] = ordered[value_col].cumsum() / total

    def label(x: float) -> str:
        if x <= 0.8:
            return "A"
        if x <= 0.95:
            return "B"
        return "C"

    ordered["abc_class"] = ordered["cum_pct"].apply(label)
    return ordered


def flag_velocity(daily: pd.DataFrame, window: int = 30, threshold: float = 1.3) -> pd.DataFrame:
    records = []
    for pid, subset in daily.groupby("product_id"):
        subset = subset.sort_values("date").set_index("date")
        rolling = subset["sales"].rolling(window=window, min_periods=1).mean()
        overall = subset["sales"].mean() or 0.0
        speed_ratio = (rolling.iloc[-1] / overall) if overall > 0 else 0.0
        label = "fast" if speed_ratio >= threshold else "slow"
        records.append({"product_id": pid, "speed_ratio": speed_ratio, "velocity_label": label})
    return pd.DataFrame(records)


def summarize_category(daily: pd.DataFrame) -> pd.DataFrame:
    copy = daily.copy()
    copy["month"] = copy["date"].dt.to_period("M").dt.to_timestamp()
    summary = (
        copy.groupby(["product_id", "month"], as_index=False)["sales"].sum()
        .rename(columns={"sales": "monthly_sales"})
    )
    return summary


def estimate_seasonality_strength(series: pd.Series, period: int = 7) -> float:
    try:
        from statsmodels.tsa.seasonal import STL
    except Exception:
        return 0.0
    if len(series) < period * 2:
        return 0.0
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    seasonal_var = np.var(result.seasonal)
    total_var = np.var(series)
    if total_var <= 0:
        return 0.0
    return float(seasonal_var / total_var)
