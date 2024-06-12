#!/bin/bash

# Cleanup data, logs, scalers, models, predictions
rm -rf data/*
rm -rf logs/*
rm -rf models/*
rm -rf predictions/*
rm -rf scalers/*

# Prepare the data
python3 src/data_preparation.py

# Setup the feature engineering
python3 src/feature_engineering.py

# Train the model
python3 src/train.py

# Predict the test data
python3 src/predict.py