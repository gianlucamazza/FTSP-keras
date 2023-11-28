# Bitcoin Price Prediction Using LSTM

## Project Overview
This project leverages a Long Short-Term Memory (LSTM) model, a variant of recurrent neural networks, to predict Bitcoin prices. It encompasses data preparation, feature engineering, model building, training, and predictions based on historical price data.

## Motivation
The goal is to explore machine learning techniques applied to financial data, aiming to develop a model capable of accurately predicting Bitcoin prices. This model could potentially aid in making informed investment decisions.

## Structure
The repository is organized into scripts for each stage of the machine learning pipeline:

- `data_preparation.py`: Processes raw Bitcoin price data.
- `feature_engineering.py`: Enhances the data with technical indicators and normalizes it.
- `model.py`: Outlines the LSTM neural network architecture.
- `train.py`: Focuses on training and evaluating the model with historical data.
- `predict.py`: Employs the trained model for future price predictions.

## Data
The data folder contains the historical Bitcoin price data:
- `BTC-USD.csv`: Original Bitcoin price dataset.
- `processed_data.csv`: Data augmented with technical indicators.
- `scaled_data.csv`: Data normalized for model training.

Processed data files:
- `processed_data.csv`: Data with added technical indicators.
- `scaled_data.csv`: Normalized data ready for model training.

## Usage
Each script is designed to be run sequentially:

1. **Data Preparation**:
   ```bash
   python data_preparation.py
    ```
2. **Feature Engineering**:
   ```bash
   python feature_engineering.py
     ```
3. **Build Model**:
   ```bash
   python model.py
     ```
4. **Train Model**:
   ```bash
   python train.py
     ```
5. **Predictions**:
6. ```bash
   python predict.py
     ```

## Requirements
The project requires the following libraries:
- Keras
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- joblib

Install dependencies using:
```bash
pip install -r requirements.txt
```


# Contributing
Contributions to improve the model or suggestions for new features are welcome. Please follow the standard pull request process to contribute.

# License
This project is open-sourced under the MIT license.
[MIT](https://choosealicense.com/licenses/mit/)