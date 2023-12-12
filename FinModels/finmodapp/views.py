# views.py
from django.shortcuts import render, redirect
import pandas as pd
import os
import matplotlib.pyplot as plt
from django.conf import settings
from .forms import CSVUploadForm
from .arima_utils import load_and_preprocess_data, build_and_fit_arima_model
from .models import CSVFile

def upload_file(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_data = pd.read_csv(request.FILES['file'])
            json_data = csv_data.to_json(orient='records')
            csv_file = form.save(commit=False)
            csv_file.content_json = json_data
            csv_file.save()
            target_col = request.POST.get('target_column')
            request.session['target_col'] = target_col  # Store the target column name in the session
            return redirect('forecast')
    else:
        form = CSVUploadForm()
    return render(request, 'finmodapp/upload.html', {'form': form})

def generate_forecast(request):
    target_col = request.session.get('target_col', 'default_column_name')
    latest_file = CSVFile.objects.latest('id')
    file_path = latest_file.file.path

    df = pd.read_csv(file_path)
    date_col = df.columns[0]  # Assume first column is the date
    df = load_and_preprocess_data(df, date_col, target_col)
    model_fit = build_and_fit_arima_model(df, target_col, (5, 1, 5))

    forecast_periods = 10
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_index = pd.date_range(df.index[-1], periods=forecast_periods + 1)
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

    plot_filename = 'forecast_plot.png'
    plot_filepath = os.path.join(settings.MEDIA_ROOT, plot_filename)
    plt.figure(figsize=(12, 6))
    plt.plot(df[target_col], label='Observed')
    plt.plot(forecast_series, label='Forecast', color='red')
    plt.title('ARIMA Model Forecast')
    plt.xlabel(date_col)
    plt.ylabel(target_col)
    plt.legend()
    plt.savefig(plot_filepath)
    plt.close()

    plot_url = os.path.join(settings.MEDIA_URL, plot_filename)
    return render(request, 'finmodapp/forecast.html', {'plot_url': plot_url})


