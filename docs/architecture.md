# Architecture Overview

This project provides an end-to-end inventory analytics pipeline designed for Kaggle execution and local development. The major components are:

## Data layer

- **Raw inputs**: `data/products.csv` and `data/transactions.csv` (or generated via `scripts/generate_synthetic.py`).
- **Update flow**: New transactional rows can be appended; the pipeline reads the full history on each run.

## Processing layer (`src/data_processing.py`)

- Load and validate input files.
- Clean missing values, enforce data types, and create daily demand series per product.
- Produce feature tables including rolling averages, cumulative sales, and current stock estimates.

## Analysis layer (`src/analysis.py`)

- Perform ABC classification using revenue contribution.
- Detect fast/slow movers through rolling demand windows.
- Estimate seasonality strength for demand series.

## Modelling layer (`src/model.py`)

- Train Prophet models per product for short-term demand forecasts.
- Optional LSTM model stub for future extension.
- Provide reorder quantity suggestions based on forecasted demand, lead time, and safety margin.

## Orchestration (`scripts/train_and_update.py`)

- CLI script orchestrates the entire pipeline (data prep → analysis → modelling → reporting).
- Saves plots, forecasts, and reorder recommendations in `outputs/`.
- Designed to be called from the Kaggle notebook for reproducible runs.

## Notebooks (`notebooks/kaggle_run.ipynb`)

- Minimal Kaggle-friendly notebook to install dependencies, run the training script, and display key outputs.

## Continuous updates

- Scheduled reruns (daily/weekly) ingest new data and regenerate forecasts.
- The modular structure makes it easy to swap models or adjust business rules without rewriting the notebook.
