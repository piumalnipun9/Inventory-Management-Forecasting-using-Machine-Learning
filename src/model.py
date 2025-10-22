"""Forecasting models and reorder recommendation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import datetime as dt


@dataclass
class ForecastResult:
    product_id: str
    forecast_df: pd.DataFrame
    total_demand_30: float


def prepare_prophet_frame(series: pd.DataFrame) -> pd.DataFrame:
    frame = series.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    frame = frame.sort_values("ds")
    return frame


def run_prophet(series: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    from prophet import Prophet

    frame = prepare_prophet_frame(series)
    model = Prophet()
    model.fit(frame)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)
    return forecast


def run_lstm_stub(series: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    # Placeholder stub for future deep learning implementation
    last_value = series.sort_values("date")["sales"].iloc[-1]
    dates = pd.date_range(series["date"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    prediction = np.full(shape=horizon_days, fill_value=last_value)
    return pd.DataFrame({"ds": dates, "yhat": prediction})


def forecast_per_product(
    daily: pd.DataFrame,
    product_ids: List[str],
    horizon_days: int,
    model_name: str,
) -> List[ForecastResult]:
    results: List[ForecastResult] = []
    for pid in product_ids:
        subset = daily[daily["product_id"] == pid][["date", "sales"]]
        if len(subset) < 5:
            continue
        if model_name == "prophet":
            forecast = run_prophet(subset, horizon_days=horizon_days)
        elif model_name == "lstm_stub":
            forecast = run_lstm_stub(subset, horizon_days=horizon_days)
        else:
            raise ValueError(f"Unknown model {model_name}")
        future = forecast.tail(horizon_days)
        total = float(future["yhat"].sum())
        results.append(ForecastResult(product_id=pid, forecast_df=forecast, total_demand_30=total))
    return results


def _compute_next_order_date(
    current_stock: float,
    reorder_level: float,
    lead_time: int,
    future_df: pd.DataFrame,
) -> Optional[dt.date]:
    today = dt.date.today()
    fut = future_df[["ds", "yhat"]].copy().reset_index(drop=True)
    fut["cum"] = fut["yhat"].clip(lower=0).cumsum()
    mask = (current_stock - fut["cum"]) <= reorder_level
    if not mask.any():
        return None
    first_idx = int(np.argmax(mask.values))
    hit_date = pd.to_datetime(fut.loc[first_idx, "ds"]).date()
    order_date = hit_date - dt.timedelta(days=int(lead_time))
    return max(today, order_date)


def suggest_reorder(
    inventory: pd.DataFrame,
    forecasts: List[ForecastResult],
    horizon_days: int,
    safety_factor: float = 1.2,
) -> pd.DataFrame:
    forecast_map: Dict[str, ForecastResult] = {f.product_id: f for f in forecasts}
    rows = []
    for _, row in inventory.iterrows():
        pid = row["product_id"]
        forecast = forecast_map.get(pid)
        if forecast is None:
            continue
        lead_time = max(int(row.get("lead_time", 7)), 1)
        current_stock = float(row.get("current_stock", 0))
        reorder_level = float(row.get("reorder_level", 0))
        # Future horizon window (assume tail is future)
        fut = forecast.forecast_df.tail(horizon_days)
        demand_during_lead = float(fut.head(lead_time)["yhat"].clip(lower=0).sum())

        # Perishable logic: limit effective demand by time until expiration
        expiration_date = row.get("expiration_date", None)
        sale_window_days: Optional[int] = None
        if pd.notna(expiration_date):
            try:
                exp = expiration_date if isinstance(expiration_date, dt.date) else pd.to_datetime(expiration_date).date()
                sale_window_days = max(0, (exp - dt.date.today()).days)
            except Exception:
                sale_window_days = None
        if sale_window_days is not None:
            effective_days = max(0, min(lead_time, sale_window_days))
            demand_during_lead = float(fut.head(effective_days)["yhat"].clip(lower=0).sum())

        reorder_qty = max(0, int(round(demand_during_lead * safety_factor - current_stock)))
        needs_reorder = (current_stock - demand_during_lead) < reorder_level

        next_order_date = _compute_next_order_date(
            current_stock=current_stock,
            reorder_level=reorder_level,
            lead_time=lead_time,
            future_df=fut,
        )

        waste_estimate = None
        if sale_window_days and sale_window_days > 0:
            demand_until_exp = float(fut.head(sale_window_days)["yhat"].clip(lower=0).sum())
            waste_estimate = max(0.0, current_stock - demand_until_exp)
        rows.append(
            {
                "product_id": pid,
                "lead_time": lead_time,
                "current_stock": current_stock,
                "projected_demand_lead_time": demand_during_lead,
                "reorder_level": reorder_level,
                "recommended_reorder_qty": reorder_qty,
                "needs_reorder": bool(needs_reorder),
                "next_order_date": next_order_date,
                "waste_estimate": waste_estimate,
            }
        )
    return pd.DataFrame(rows)


def save_forecast_plots(
    forecasts: List[ForecastResult],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for result in forecasts:
        plt.figure(figsize=(9, 4))
        plt.plot(result.forecast_df["ds"], result.forecast_df["yhat"], label="Forecast")
        if "yhat_lower" in result.forecast_df.columns:
            plt.fill_between(
                result.forecast_df["ds"],
                result.forecast_df["yhat_lower"],
                result.forecast_df["yhat_upper"],
                color="lightblue",
                alpha=0.4,
            )
        plt.title(f"Forecast for {result.product_id}")
        plt.xlabel("Date")
        plt.ylabel("Units")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"forecast_{result.product_id}.png")
        plt.close()
