import sys
import argparse
from pathlib import Path
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from typing import List

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.config import COLUMN_SETS, CLOSE
from src.logging.logger import setup_logger

# Setup logger
ROOT_DIR = project_dir
logger = setup_logger('feature_selection_logger', 'logs', 'feature_selection.log')


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path, index_col='Date', parse_dates=True)


def evaluate_model(X, y, num_features: int) -> float:
    """Evaluate the model using cross-validation and return the mean score."""
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()  # Convert to positive
    return mean_score


def find_optimal_num_features(X, y, max_features: int) -> int:
    """Find the optimal number of features using RFE."""
    best_num_features = 1
    best_score = float('inf')
    for num_features in range(1, max_features + 1):
        score = evaluate_model(X, y, num_features)
        logger.info(f"Number of features: {num_features}, Score: {score}")
        if score < best_score:
            best_score = score
            best_num_features = num_features
    logger.info(f"Optimal number of features: {best_num_features}")
    return best_num_features


def main(ticker: str) -> None:
    """Main function to find the optimal number of features for a given ticker."""
    logger.info(f"Finding optimal number of features for {ticker}.")
    file_path = ROOT_DIR / f'data/processed_data_{ticker}.csv'
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    df = load_data(file_path)
    X = df.drop(columns=[CLOSE])
    y = df[CLOSE]

    max_features = X.shape[1]
    optimal_num_features = find_optimal_num_features(X, y, max_features)
    logger.info(f"Optimal number of features for {ticker}: {optimal_num_features}")

    # Save the optimal number of features to a file
    optimal_features_path = ROOT_DIR / f'{ticker}_optimal_features_.txt'
    with open(optimal_features_path, 'w') as f:
        f.write(str(optimal_num_features))
    logger.info(f"Optimal number of features saved to {optimal_features_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Selection')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker')
    args = parser.parse_args()
    main(ticker=args.ticker)
