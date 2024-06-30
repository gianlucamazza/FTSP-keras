from setuptools import setup, find_packages

setup(
    name='lstm_price_prediction',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'featuretools',
        'joblib',
        'keras',
        'matplotlib',
        'numpy',
        'pmdarima',
        'optuna',
        'optuna_integration',
        'pandas',
        'PyQt5',
        'scikit-learn',
        'statsmodels',
        'tensorflow',
        'tensorrt',
        'yfinance',
        'tensorflow-metadata',
    ],
)
