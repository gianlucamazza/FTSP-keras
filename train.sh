#!/bin/bash

# Create a conda environment
conda create --name myenv python=3.11 -y

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv

# Configure conda to use the conda-forge channel
conda config --add channels conda-forge
conda config --set channel_priority strict

# Install dependencies from conda-forge
conda install -y pandas numpy tensorflow featuretools statsmodels matplotlib scikit-learn keras joblib yfinance pyqt optuna optuna-integration tensorflow-cloud

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
