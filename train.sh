#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run cleanup.sh
bash cleanup.sh

# Prepare the data
python3 src/data_preparation.py

# Setup the feature engineering
python3 src/feature_engineering.py

# Train the model
python3 src/train.py

# Deactivate the virtual environment
deactivate
