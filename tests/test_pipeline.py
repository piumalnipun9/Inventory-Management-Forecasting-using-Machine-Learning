"""Smoke tests for the inventory pipeline."""
from __future__ import annotations

from pathlib import Path

from scripts.generate_synthetic import main as generate_synthetic
from src import pipeline


def test_pipeline_runs_with_synthetic(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    generate_synthetic(out_dir=data_dir, n_products=10, start="2024-01-01", end="2024-03-31", seed=7)

    results = pipeline.run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name="lstm_stub",
        horizon_days=14,
        plot_examples=1,
    )

    assert not results.abc.empty
    assert not results.velocity.empty
    assert not results.seasonality.empty
    assert not results.reorder.empty
    assert any(output_dir.glob("plots/forecast_*.png"))
