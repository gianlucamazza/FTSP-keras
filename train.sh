#!/bin/bash

# Check if running in Bash
if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run with Bash."
  exit 1
fi

# Set ticker, date range, and names
TICKER_DATES=(
  "GC=F,2000-01-01,GOLDUSD,B"
  "BTC-USD,2010-01-01,BTCUSD,D"
)

# Run cleanup script
bash cleanup.sh || { echo "Cleanup failed"; exit 1; }

# Loop through each ticker
for ENTRY in "${TICKER_DATES[@]}"; do
  IFS=',' read -r TICKER START_DATE NAME FREQ <<< "$ENTRY"
  echo "Processing NAME"

  # Prepare the data
  python src/data/data_preparation.py --yfinance_ticker="$TICKER" --ticker="$NAME" --start_date="$START_DATE" || { echo "Data preparation failed for $TICKER"; exit 1; }

  # Calculate indicators
  python src/data/calculate_indicators.py --ticker="$NAME" || { echo "Indicator calculation failed for $TICKER"; exit 1; }

  # Setup feature engineering
  python src/data/feature_engineering.py --ticker="$NAME" --freq="$FREQ" || { echo "Feature engineering failed for $TICKER"; exit 1; }

  # Train the model
  python src/training/train_model.py --ticker="$NAME" || { echo "Model training failed for $TICKER"; exit 1; }
done
