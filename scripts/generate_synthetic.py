"""Generate a synthetic inventory dataset for experimentation."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd


CATEGORIES = [
    "Electronics",
    "Apparel",
    "Home",
    "Grocery",
    "Beauty",
]


def build_products(n_products: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for idx in range(1, n_products + 1):
        category = rng.choice(CATEGORIES)
        lead_time = int(rng.integers(2, 21))
        reorder_level = int(rng.integers(20, 180))
        initial_stock = int(rng.integers(100, 1500))
        unit_cost = float(np.round(rng.uniform(3, 350), 2))
        records.append(
            {
                "product_id": f"P{idx:04d}",
                "category": category,
                "lead_time": lead_time,
                "reorder_level": reorder_level,
                "initial_stock": initial_stock,
                "unit_cost": unit_cost,
            }
        )
    return pd.DataFrame(records)


def seasonal_multiplier(dates: pd.DatetimeIndex, category: str) -> np.ndarray:
    # simple seasonal pattern by category
    base = np.ones(len(dates))
    month = dates.month
    if category == "Electronics":
        base += np.where((month >= 10) | (month <= 1), 0.6, 0.0)
    elif category == "Apparel":
        base += np.where((month >= 3) & (month <= 5), 0.4, 0.0)
    elif category == "Grocery":
        base += np.where(month == 12, 0.3, 0.1)
    elif category == "Beauty":
        base += np.where(month == 2, 0.2, 0.0)
    return base


def build_transactions(
    products: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    rows = []
    for _, prod in products.iterrows():
        base_demand = max(2.0, 800.0 / (prod["unit_cost"] + 10.0))
        weekly_pattern = 1.0 + 0.2 * np.sin(np.arange(len(dates)) * (2 * np.pi / 7))
        seasonal = seasonal_multiplier(dates, prod["category"])
        noise = rng.normal(loc=1.0, scale=0.2, size=len(dates))
        demand = base_demand * weekly_pattern * seasonal * noise
        demand = np.clip(demand, a_min=0.0, a_max=None)
        demand = np.round(demand).astype(int)
        for dt_idx, qty in zip(dates, demand):
            if qty <= 0:
                continue
            rows.append(
                {
                    "date": dt_idx.date().isoformat(),
                    "product_id": prod["product_id"],
                    "category": prod["category"],
                    "sales": int(qty),
                }
            )
    return pd.DataFrame(rows)


def main(
    out_dir: Path,
    n_products: int,
    start: str,
    end: str,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    products = build_products(n_products=n_products, seed=seed)
    transactions = build_transactions(
        products=products,
        start=dt.date.fromisoformat(start),
        end=dt.date.fromisoformat(end),
        seed=seed + 7,
    )
    products.to_csv(out_dir / "products.csv", index=False)
    transactions.to_csv(out_dir / "transactions.csv", index=False)
    print(
        f"Generated {len(products)} products and {len(transactions)} transactions into {out_dir}"  # noqa: T201
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic inventory dataset")
    parser.add_argument("--out-dir", default="data", help="Output directory for CSV files")
    parser.add_argument("--products", type=int, default=80, help="Number of products")
    parser.add_argument("--start", default="2023-01-01", help="Start date (ISO format)")
    parser.add_argument("--end", default="2024-12-31", help="End date (ISO format)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(
        out_dir=Path(args.out_dir),
        n_products=args.products,
        start=args.start,
        end=args.end,
        seed=args.seed,
    )
