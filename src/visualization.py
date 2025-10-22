"""Visualization helpers for analysis outputs."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_style("whitegrid")


def plot_stock_vs_sales(inventory: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=inventory,
        x="current_stock",
        y="total_sales_to_date",
        hue="category",
    )
    plt.title("Current Stock vs Annual Sales")
    plt.xlabel("Current Stock")
    plt.ylabel("Historical Sales")
    plt.tight_layout()
    path = out_dir / "stock_vs_sales.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_abc_pie(abc_df: pd.DataFrame, out_dir: Path) -> Path:
    counts = abc_df["abc_class"].value_counts().sort_index()
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title("ABC Classification Share")
    path = out_dir / "abc_distribution.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_monthly_trend(monthly: pd.DataFrame, product_id: str, out_dir: Path) -> Path:
    subset = monthly[monthly["product_id"] == product_id]
    if subset.empty:
        raise ValueError(f"No monthly data for product {product_id}")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=subset, x="month", y="monthly_sales")
    plt.title(f"Monthly Demand Trend - {product_id}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = out_dir / f"monthly_trend_{product_id}.png"
    plt.savefig(path)
    plt.close()
    return path
