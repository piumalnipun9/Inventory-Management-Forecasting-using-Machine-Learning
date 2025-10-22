"""Data loading and preprocessing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd


EXPECTED_PRODUCT_COLUMNS = {
    "product_id",
    "category",
    "lead_time",
    "reorder_level",
    "initial_stock",
    "unit_cost",
}

EXPECTED_TRANSACTION_COLUMNS = {
    "date",
    "product_id",
    "category",
    "sales",
}


def load_products(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Products file not found: {path}")
    df = pd.read_csv(path)
    missing = EXPECTED_PRODUCT_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Products file missing columns: {sorted(missing)}")
    df["lead_time"] = df["lead_time"].fillna(df["lead_time"].median()).astype(int)
    df["reorder_level"] = df["reorder_level"].fillna(df["reorder_level"].median()).astype(int)
    df["initial_stock"] = df["initial_stock"].fillna(df["initial_stock"].median()).astype(int)
    df["unit_cost"] = df["unit_cost"].fillna(df["unit_cost"].median()).astype(float)
    return df


def load_transactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Transactions file not found: {path}")
    df = pd.read_csv(path)
    missing = EXPECTED_TRANSACTION_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Transactions file missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "product_id"])
    df["sales"] = df["sales"].fillna(0).astype(int)
    df = df[df["sales"] >= 0]
    return df


def aggregate_daily_sales(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = (
        df.groupby(["product_id", "date"], as_index=False)["sales"]
        .sum()
        .sort_values(["product_id", "date"])
    )
    return daily


def add_rolling_features(daily: pd.DataFrame, windows=(7, 14, 30)) -> pd.DataFrame:
    frames = []
    for pid, subset in daily.groupby("product_id", sort=False):
        s = subset.sort_values("date").set_index("date")
        for win in windows:
            s[f"rolling_{win}"] = (
                s["sales"].rolling(window=win, min_periods=1).mean()
            )
        s["cumulative"] = s["sales"].cumsum()
        s["lag_1"] = s["sales"].shift(1)
        s = s.reset_index()
        s["product_id"] = pid
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def compute_current_stock(
    products_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    totals = transactions_df.groupby("product_id", as_index=False)["sales"].sum()
    totals = totals.rename(columns={"sales": "total_sales_to_date"})
    merged = products_df.merge(totals, on="product_id", how="left")
    merged["total_sales_to_date"] = merged["total_sales_to_date"].fillna(0)
    merged["current_stock"] = merged["initial_stock"] - merged["total_sales_to_date"]
    merged["current_stock"] = merged["current_stock"].clip(lower=0)
    return merged


def prepare_datasets(
    products_path: Path,
    transactions_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products = load_products(products_path)
    tx = load_transactions(transactions_path)
    daily = aggregate_daily_sales(tx)
    daily_features = add_rolling_features(daily)
    inventory = compute_current_stock(products, tx)
    return products, daily_features, inventory



