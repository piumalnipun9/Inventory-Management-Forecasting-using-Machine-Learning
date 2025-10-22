"""CLI entry-point to run the inventory management pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from src import pipeline
from scripts.generate_synthetic import main as generate_synthetic


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.generate_synthetic or not (data_dir / "transactions.csv").exists():
        print("Generating synthetic dataset...")  # noqa: T201
        generate_synthetic(
            out_dir=data_dir,
            n_products=args.synthetic_products,
            start=args.synthetic_start,
            end=args.synthetic_end,
            seed=args.synthetic_seed,
        )

    pipeline_outputs = pipeline.run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=args.model,
        horizon_days=args.horizon,
        plot_examples=args.plot_examples,
    )

    print(f"ABC classes saved at {output_dir / 'abc_classification.csv'}")  # noqa: T201
    print(f"Velocity metrics saved at {output_dir / 'velocity_metrics.csv'}")  # noqa: T201
    print(f"Reorder recommendations saved at {output_dir / 'reorder_recommendations.csv'}")  # noqa: T201
    print(f"Forecast plots stored in {(output_dir / 'plots').resolve()}")  # noqa: T201
    print(f"Processed {len(pipeline_outputs.forecasts)} products for forecasting")  # noqa: T201


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inventory forecasting pipeline")
    parser.add_argument("--data-dir", default="data", help="Directory containing products.csv & transactions.csv")
    parser.add_argument("--output-dir", default="outputs", help="Directory to store results")
    parser.add_argument("--model", default="prophet", choices=["prophet", "lstm_stub"], help="Forecasting model")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    parser.add_argument("--plot-examples", type=int, default=3, help="Number of monthly trend plots to create")

    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic dataset before running")
    parser.add_argument("--synthetic-products", type=int, default=80, help="Number of synthetic products")
    parser.add_argument("--synthetic-start", default="2023-01-01", help="Synthetic data start date")
    parser.add_argument("--synthetic-end", default="2024-12-31", help="Synthetic data end date")
    parser.add_argument("--synthetic-seed", type=int, default=42, help="Synthetic generation seed")
    return parser


if __name__ == "__main__":
    cli_parser = build_parser()
    run(cli_parser.parse_args())
