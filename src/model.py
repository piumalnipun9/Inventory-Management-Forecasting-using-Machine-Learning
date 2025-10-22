"""Forecasting models and reorder recommendation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


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
        demand_during_lead = forecast.total_demand_30 * (lead_time / horizon_days)
        reorder_qty = max(0, int(round(demand_during_lead * safety_factor - current_stock)))
        needs_reorder = (current_stock - demand_during_lead) < reorder_level
        rows.append(
            {
                "product_id": pid,
                "lead_time": lead_time,
                "current_stock": current_stock,
                "projected_demand_lead_time": demand_during_lead,
                "reorder_level": reorder_level,
                "recommended_reorder_qty": reorder_qty,
                "needs_reorder": bool(needs_reorder),
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
