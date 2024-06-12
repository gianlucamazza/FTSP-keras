#!/bin/bash

# Run cleanup.sh
bash cleanup.sh

# Prepare the data
python3 src/data_preparation.py

# Setup the feature engineering
python3 src/feature_engineering.py

# Train the model
python3 src/train.py

# Predict the test data
python3 src/predict.py