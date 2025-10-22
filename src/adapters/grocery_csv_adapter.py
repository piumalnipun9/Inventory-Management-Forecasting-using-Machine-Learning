"""Adapter to convert Grocery_Inventory_new_v1.csv into pipeline inputs.

Expected input columns:
- Product_Name, Catagory, Supplier_Name, Warehouse_Location, Status,
  Product_ID, Supplier_ID, Date_Received, Last_Order_Date, Expiration_Date,
  Stock_Quantity, Reorder_Level, Reorder_Quantity, Unit_Price, Sales_Volume,
  Inventory_Turnover_Rate, percentage

Outputs:
- products.csv with: product_id, category, lead_time, reorder_level, initial_stock,
  unit_cost, expiration_date, current_stock
- transactions.csv with: date, product_id, category, sales (synthetic daily demand)
"""
from __future__ import annotations

from pathlib import Path
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd


def _parse_money(val: object) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def _safe_date(s: object) -> Optional[dt.date]:
    if pd.isna(s):
        return None
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None


def convert_grocery_csv_to_internal(input_csv: Path, out_dir: Path, default_lead_time: int = 7) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    # Normalize column names
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    # Some datasets use 'Catagory' (typo); normalize to 'Category'
    if "Catagory" in df.columns:
        df = df.rename(columns={"Catagory": "Category"})

    # Parse fields
    df["Date_Received_parsed"] = df["Date_Received"].apply(_safe_date)
    df["Last_Order_Date_parsed"] = df["Last_Order_Date"].apply(_safe_date)
    df["Expiration_Date_parsed"] = df["Expiration_Date"].apply(_safe_date)
    df["Unit_Cost"] = df["Unit_Price"].apply(_parse_money)

    # Build products.csv
    products = pd.DataFrame({
        "product_id": df["Product_ID"].astype(str),
        "category": df.get("Category", df.get("Catagory", "Grocery")).astype(str),
        "lead_time": df.get("Lead_Time", default_lead_time),
        "reorder_level": pd.to_numeric(df.get("Reorder_Level", 0), errors="coerce").fillna(0).astype(int),
        "initial_stock": pd.to_numeric(df.get("Stock_Quantity", 0), errors="coerce").fillna(0).astype(int),
        "unit_cost": df["Unit_Cost"].astype(float),
        "expiration_date": df["Expiration_Date_parsed"].astype("datetime64[ns]").dt.date,
        "current_stock": pd.to_numeric(df.get("Stock_Quantity", 0), errors="coerce").fillna(0).astype(int),
    })
    products.to_csv(out_dir / "products.csv", index=False)

    # Build transactions.csv with synthetic daily demand matching Sales_Volume
    rows = []
    today = dt.date.today()
    rng = np.random.default_rng(42)
    for _, r in df.iterrows():
        pid = str(r["Product_ID"])
        cat = str(r.get("Category", r.get("Catagory", "Grocery")))
        start = r.get("Date_Received_parsed") or (today - dt.timedelta(days=90))
        end = r.get("Last_Order_Date_parsed") or today
        if start > end:
            start, end = end, start
        dates = pd.date_range(start=start, end=end, freq="D")
        total_sales = int(pd.to_numeric(r.get("Sales_Volume", 0), errors="coerce") or 0)
        if len(dates) == 0 or total_sales <= 0:
            continue
        # Weekday profile (more sales on weekends for grocery?)
        weekday = np.array([d.weekday() for d in dates])
        base = np.ones(len(dates))
        base += np.where(weekday >= 5, 0.3, 0.0)  # Sat/Sun +30%
        base = base / base.sum()
        # Allocate integer sales across days
        alloc = np.floor(base * total_sales).astype(int)
        remainder = total_sales - int(alloc.sum())
        if remainder > 0:
            idxs = rng.choice(len(dates), size=remainder, replace=True)
            for i in idxs:
                alloc[i] += 1
        for d, q in zip(dates, alloc):
            if q <= 0:
                continue
            rows.append({
                "date": d.date().isoformat(),
                "product_id": pid,
                "category": cat,
                "sales": int(q),
            })
    tx = pd.DataFrame(rows)
    if not tx.empty:
        tx.to_csv(out_dir / "transactions.csv", index=False)
    else:
        # create minimal empty file to avoid errors downstream
        pd.DataFrame(columns=["date", "product_id", "category", "sales"]).to_csv(out_dir / "transactions.csv", index=False)
