#!/bin/bash

# Set ticker and date range
TICKERS=("GLD" "BTC")
START_DATE="2015-01-01"
END_DATE="2024-06-29"

# Run cleanup script
bash cleanup.sh

# Loop through each ticker
for TICKER in "${TICKERS[@]}"; do
  echo "Processing $TICKER"

  # Model params
  PARAMS="${TICKER}_best_params.json"

  # Prepare the data
  python src/data_preparation.py --ticker="$TICKER" --start_date="$START_DATE" --end_date="$END_DATE"

  # Setup feature engineering
  python src/feature_engineering.py --ticker="$TICKER"

  # Train the model
  python src/train.py --ticker="$TICKER" --best_params="$PARAMS"
done
