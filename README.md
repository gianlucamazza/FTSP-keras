# Bitcoin Price Prediction Using LSTM

## Project Overview
This project focuses on predicting Bitcoin prices by analyzing historical price data using a Long Short-Term Memory (LSTM) model, a type of recurrent neural network. The project involves data preparation, feature engineering, model training, and prediction.

## Structure
The project is organized into several scripts, each handling a specific aspect of the machine learning workflow:

- `data_preparation.py`: Loads and preprocesses the raw Bitcoin price data.
- `feature_engineering.py`: Adds technical indicators as features and normalizes the data.
- `model.py`: Defines the LSTM model architecture.
- `train.py`: Trains the model on historical data and evaluates its performance.
- `predict.py`: Uses the trained model to make future price predictions.

## Data
The data folder contains the historical Bitcoin price data:
- `BTC-USD.csv`: Raw Bitcoin price data.

Processed data files:
- `processed_data.csv`: Data with added technical indicators.
- `scaled_data.csv`: Normalized data ready for model training.

## Usage
Each script is designed to be run sequentially:

1. **Data Preparation**:
   ```bash
   python src/data_preparation.py
    ```
2. **Feature Engineering**:
   ```bash
    python src/feature_engineering.py
     ```
   
3. **Model Training**:
    ```bash
     python src/train.py
      ```
4. **Prediction**:
    ```bash
     python src/predict.py
      ```

## Requirements
- Keras
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- joblib

```bash
pip install -r requirements.txt
```

# Note
This project is intended for educational purposes and should not be used as financial advice.

# License
[MIT](https://choosealicense.com/licenses/mit/)