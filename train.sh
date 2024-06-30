#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run with Bash."
  exit 1
fi

# Set ticker, date range, and names
declare -A TICKER_DATES=(
  ["GC=F"]="2000-01-01,GOLD"
  ["BTC-USD"]="2010-01-01,BTC"
)

# Run cleanup script
bash cleanup.sh

# Loop through each ticker
for TICKER in "${!TICKER_DATES[@]}"; do
  echo "Processing $TICKER"

  IFS=',' read -r START_DATE NAME <<< "${TICKER_DATES[$TICKER]}"

  # Prepare the data
  python src/data/data_preparation.py --ticker="$TICKER"  --name="$NAME" --start_date="$START_DATE"

  # Setup feature engineering
  python src/data/feature_engineering.py --ticker="$TICKER" --name="$NAME"

  # Train the model
  python src/training/train_model.py --ticker="$TICKER" --name="$NAME"
done
