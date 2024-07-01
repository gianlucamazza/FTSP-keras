#!/bin/bash

# Check if running in Bash
if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run with Bash."
  exit 1
fi

# Set ticker, date range, and names
TICKER_DATES=(
  "BTC-USD,2020-01-01,BTCUSD,D"
#  "GC=F,2000-01-01,GOLDUSD,B"
)

# Run cleanup script
bash cleanup.sh || { echo "Cleanup failed"; exit 1; }

# Loop through each ticker
for ENTRY in "${TICKER_DATES[@]}"; do
  IFS=',' read -r TICKER START_DATE NAME FREQ <<< "$ENTRY"
  echo "Processing $NAME"

  # Prepare the data
  python src/data/data_preparation.py --yfinance_ticker="$TICKER" --ticker="$NAME" --start_date="$START_DATE" || { echo "Data preparation failed for $NAME"; exit 1; }

  # Calculate indicators
  python src/data/calculate_indicators.py --ticker="$NAME" || { echo "Indicator calculation failed for $NAME"; exit 1; }

  # Find the optimal number of features
  python src/data/find_optimal_features.py --ticker="$NAME" || { echo "Finding optimal features failed for $NAME"; exit 1; }

  # Read the optimal number of features
  OPTIMAL_FEATURES=$(cat "${NAME}_optimal_features.txt")

  # Setup feature engineering
  python src/data/feature_engineering.py --ticker="$NAME" --freq="$FREQ" --num_features="$OPTIMAL_FEATURES" || { echo "Feature engineering failed for $NAME"; exit 1; }

  # Train the model
  # python src/training/objective.py --ticker="$NAME" || { echo "Objective failed for $NAME"; exit 1; }

  # Train the model
  # python src/training/train_model.py --ticker="$NAME" || { echo "Model training failed for $NAME"; exit 1; }
done
