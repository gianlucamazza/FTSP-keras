import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.logging.logger import setup_logger

ROOT_DIR = Path(__file__).parent.parent
logger = setup_logger('eda', 'logs', 'eda.log')

def load_data(file_path: str):
    """Load the dataset from the specified path."""
    logger.info(f"Loading data from {file_path}.")
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

def missing_values_analysis(df):
    """Analyze missing values in the dataset."""
    logger.info("Analyzing missing values.")
    missing_values = df.isnull().mean()
    logger.info(f"Missing values analysis completed:\n{missing_values}")
    return missing_values

def variance_analysis(df, threshold=0.01):
    """Analyze variance of columns in the dataset."""
    logger.info("Analyzing variance of columns.")
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    variances = df.var()
    low_variance_cols = variances[variances < threshold].index.tolist()
    logger.info(f"Columns with variance below {threshold}:\n{low_variance_cols}")
    return low_variance_cols

def correlation_analysis(df, target_col):
    """Analyze correlation of columns with the target variable."""
    logger.info(f"Analyzing correlation with target column '{target_col}'.")
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    correlation_matrix = df.corr()
    target_correlation = correlation_matrix[target_col].sort_values(ascending=False)
    logger.info(f"Correlation with target column '{target_col}':\n{target_correlation}")
    return target_correlation

def plot_distributions(df, cols):
    """Plot distributions of the specified columns."""
    logger.info("Plotting distributions of columns.")
    for col in cols:
        if col != 'Date':
            logger.info(f"Plotting distribution for column '{col}'.")
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()
    logger.info("Distribution plots completed.")

def eda_pipeline(file_path, target_col):
    """Complete pipeline for EDA."""
    logger.info("Starting EDA pipeline.")
    try:
        df = load_data(file_path)
        missing_values = missing_values_analysis(df)
        low_variance_cols = variance_analysis(df)
        target_correlation = correlation_analysis(df, target_col)
        plot_distributions(df, [col for col in df.columns if col != 'Date'])
        logger.info("EDA pipeline completed successfully.")
        return df, missing_values, low_variance_cols, target_correlation
    except Exception as e:
        logger.error(f"EDA pipeline failed: {e}")
        raise
