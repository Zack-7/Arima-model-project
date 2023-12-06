from django.db import models
from django.db.models import JSONField

class CSVFile(models.Model):
    file = models.FileField(upload_to='csv_files/')
    content_json = JSONField(blank=True, null=True)

def __str__(self):
        return f"{self.file.name} - {self.pk}"


import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

if __name__ == "__main__":
    file_path = 'C:\\Users\\Onke Gwala\\datasets\\dailydeli.csv'  
    date_col = 'date'  # Replace with the actual date column name
    target_col = 'meantemp'  # Replace with the actual target column name
    order = (5, 1, 5) 

def load_and_preprocess_data(file_path, date_col, target_col):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Convert to datetime and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    df = df.asfreq('D')

    # Differencing to make data stationary if necessary
    result = adfuller(df[target_col])
    if result[1] > 0.05:  
        df[target_col] = df[target_col].diff().dropna()

    return df

def build_and_fit_arima_model(dataframe, target_col, order):
    # Build and fit the ARIMA model
    model = ARIMA(dataframe[target_col], order=order)
    model_fit = model.fit()

    return model_fit

df = load_and_preprocess_data(file_path, date_col, target_col)
model_fit = build_and_fit_arima_model(df, target_col, order)

import matplotlib.pyplot as plt

forecast_periods = 10
forecast = model_fit.get_forecast(steps=forecast_periods)
forecast_index = pd.date_range(df.index[-1], periods=forecast_periods + 1)
forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

plt.figure(figsize=(12, 6))
plt.plot(df[target_col], label='Observed')
plt.plot(forecast_series, label='Forecast', color='red')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.legend()
plt.show()