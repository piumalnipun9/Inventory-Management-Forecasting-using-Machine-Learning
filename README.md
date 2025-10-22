# Inventory Management Analysis & Prediction

This repository delivers an end-to-end pipeline that ingests inventory transactions, performs ABC and velocity analysis, forecasts demand (Prophet or LSTM stub), and produces reorder recommendations together with Canva-ready visuals.

## Repository layout

- `src/` – core modules: data prep, analytics, forecasting, plotting, and high-level pipeline.
- `scripts/` – CLI utilities (`generate_synthetic.py`, `train_and_update.py`).
- `notebooks/` – Kaggle-friendly notebook to run the pipeline end-to-end.
- `docs/` – architecture notes and the Canva presentation outline.
- `data/` – expected location for `products.csv` and `transactions.csv` (auto-created).
- `models/`, `outputs/` – generated forecasts, plots, and CSV exports.

## Data schema

`products.csv`

| column          | description                              |
| --------------- | ---------------------------------------- |
| `product_id`    | unique SKU identifier                    |
| `category`      | product category (string)                |
| `lead_time`     | supplier lead time in days               |
| `reorder_level` | business-defined minimum stock threshold |
| `initial_stock` | opening stock quantity                   |
| `unit_cost`     | unit cost/value                          |

`transactions.csv`

| column       | description                 |
| ------------ | --------------------------- |
| `date`       | ISO date of the transaction |
| `product_id` | SKU reference               |
| `category`   | category label              |
| `sales`      | daily quantity sold         |

## Quick start (local)

1. `python -m venv .venv && .venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. `python scripts/train_and_update.py --generate-synthetic`
4. Inspect outputs in `outputs/` (plots and CSV reports).

## Kaggle workflow

1. Upload this repository (or copy key files) into a Kaggle Notebook.
2. Place the dataset under `/kaggle/input/...` (or set `USE_SYNTHETIC=True` inside `notebooks/kaggle_run.ipynb`).
3. Run notebook cells sequentially to regenerate forecasts and export visuals for Canva.

### Optional: auto-sync with GitHub Actions

The workflow in `.github/workflows/kaggle-sync.yml` can publish the repository to a Kaggle Dataset whenever `main` changes:

1. Create a Kaggle API token (Account → Create New API Token) and add the values as GitHub repository secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`.
2. Update `kaggle/dataset-metadata.json` with the dataset `id` you own (format `username/dataset-name`).
3. Ensure the dataset already exists on Kaggle (create it once manually, uploading the same metadata file). After that, every push to `main` will run the action and call `kaggle datasets version ...` to upload a fresh zip containing `src/`, `scripts/`, `notebooks/`, `docs/`, and the README.

Limitations:

- Kaggle Notebooks cannot be auto-run from GitHub; the action only updates the dataset package.
- If the action fails, check the workflow logs for Kaggle CLI errors (often due to missing secrets or wrong dataset id).

## Outputs

- `abc_classification.csv` – ABC tagging with revenue contribution.
- `velocity_metrics.csv` – fast/slow mover labels and speed ratios.
- `seasonality_strength.csv` – STL-based weekly seasonality scores per product.
- `reorder_recommendations.csv` – demand during lead time, reorder quantities, and priority flags.
- `outputs/plots/` – PNG charts (stock vs sales, ABC pie, monthly trends, per-product forecasts).

## Architecture & presentation

- `docs/architecture.md` – component overview and update flow.
- `docs/presentation_outline.md` – 12-slide Canva structure with key visuals to export.

## Testing

Run `pytest` (see `tests/` folder) to validate the pipeline on synthetic data using the LSTM stub for speed.
