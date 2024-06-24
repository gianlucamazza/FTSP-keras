#!/bin/bash

# Run cleanup.sh
bash cleanup.sh

# Prepare the data
python src/data_preparation.py

# Setup the feature engineering
python src/feature_engineering.py

# Train the model
python src/train.py

# Deactivate the conda environment
conda deactivate
