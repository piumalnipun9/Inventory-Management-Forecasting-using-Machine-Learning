"""High-level orchestration pipeline for inventory analysis and forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from . import analysis, data_processing, model, visualization


@dataclass
class PipelineOutputs:
    products: pd.DataFrame
    daily: pd.DataFrame
    inventory: pd.DataFrame
    abc: pd.DataFrame
    velocity: pd.DataFrame
    seasonality: pd.DataFrame
    forecasts: List[model.ForecastResult]
    reorder: pd.DataFrame
    plots: Dict[str, Path]


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    model_name: str = "prophet",
    horizon_days: int = 30,
    plot_examples: int = 3,
) -> PipelineOutputs:
    products_path = data_dir / "products.csv"
    transactions_path = data_dir / "transactions.csv"

    products, daily_features, inventory = data_processing.prepare_datasets(
        products_path=products_path,
        transactions_path=transactions_path,
    )

    revenue = analysis.compute_revenue(products, daily_features)
    abc = analysis.abc_classification(revenue)
    velocity = analysis.flag_velocity(daily_features)
    seasonality_records = []
    for pid, subset in daily_features.groupby("product_id"):
        strength = analysis.estimate_seasonality_strength(
            subset.sort_values("date")["sales"]
        )
        seasonality_records.append(
            {"product_id": pid, "seasonality_strength": strength}
        )
    seasonality_df = pd.DataFrame(seasonality_records)
    monthly = analysis.summarize_category(daily_features)

    focus_products = abc["product_id"].head(20).tolist()
    forecasts = model.forecast_per_product(
        daily=daily_features,
        product_ids=focus_products,
        horizon_days=horizon_days,
        model_name=model_name,
    )

    reorder = model.suggest_reorder(
        inventory=inventory,
        forecasts=forecasts,
        horizon_days=horizon_days,
    )

    plot_dir = output_dir / "plots"
    inventory_stats = inventory.merge(
        revenue[["product_id", "annual_revenue"]],
        on="product_id",
        how="left",
    )

    plots: Dict[str, Path] = {}
    plots["stock_vs_sales"] = visualization.plot_stock_vs_sales(
        inventory=inventory_stats,
        out_dir=plot_dir,
    )
    plots["abc_pie"] = visualization.plot_abc_pie(abc_df=abc, out_dir=plot_dir)
    for product_id in focus_products[:plot_examples]:
        try:
            plots[f"monthly_{product_id}"] = visualization.plot_monthly_trend(
                monthly=monthly,
                product_id=product_id,
                out_dir=plot_dir,
            )
        except ValueError:
            continue
    model.save_forecast_plots(forecasts=forecasts, out_dir=plot_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    abc.to_csv(output_dir / "abc_classification.csv", index=False)
    velocity.to_csv(output_dir / "velocity_metrics.csv", index=False)
    seasonality_df.to_csv(output_dir / "seasonality_strength.csv", index=False)
    reorder.to_csv(output_dir / "reorder_recommendations.csv", index=False)

    return PipelineOutputs(
        products=products,
        daily=daily_features,
        inventory=inventory,
        abc=abc,
        velocity=velocity,
        seasonality=seasonality_df,
        forecasts=forecasts,
        reorder=reorder,
        plots=plots,
    )
