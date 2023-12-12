import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def load_and_preprocess_data(df, date_col, target_col):
    # Convert to datetime and set as index
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    # Differencing to make data stationary if necessary
    result = adfuller(df[target_col])
    if result[1] > 0.05:
        df[target_col] = df[target_col].diff().dropna()

    return df

def build_and_fit_arima_model(dataframe, target_col, order):
    model = ARIMA(dataframe[target_col], order=order)
    model_fit = model.fit()
    return model_fit
