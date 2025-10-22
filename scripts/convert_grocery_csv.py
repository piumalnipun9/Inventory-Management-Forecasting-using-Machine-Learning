"""CLI tool to convert Grocery_Inventory_new_v1.csv to internal dataset files."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.grocery_csv_adapter import convert_grocery_csv_to_internal


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert grocery CSV to products.csv and transactions.csv")
    parser.add_argument("--input", required=True, help="Path to Grocery_Inventory_new_v1.csv")
    parser.add_argument("--out-dir", default="data", help="Output directory for converted CSVs")
    parser.add_argument("--lead-time", type=int, default=7, help="Default lead time in days")
    args = parser.parse_args()

    convert_grocery_csv_to_internal(
        input_csv=Path(args.input),
        out_dir=Path(args.out_dir),
        default_lead_time=args.lead_time,
    )
    print(f"Converted dataset saved to {Path(args.out_dir).resolve()}")  # noqa: T201


if __name__ == "__main__":
    main()
