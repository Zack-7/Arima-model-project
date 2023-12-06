from django.shortcuts import render
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from django.conf import settings
from .forms import CSVUploadForm
from .models import load_and_preprocess_data, build_and_fit_arima_model

def upload_file(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Process the file and build ARIMA model
            date_col = 'date'  # Adjust as needed
            target_col = 'meantemp'  # Adjust as needed
            df = pd.read_csv(file_path)
            df = load_and_preprocess_data(df, date_col, target_col)
            model_fit = build_and_fit_arima_model(df, target_col, (5, 1, 5))

            # Generate forecast
            forecast_periods = 10
            forecast = model_fit.get_forecast(steps=forecast_periods)
            forecast_index = pd.date_range(df.index[-1], periods=forecast_periods + 1)
            forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

            # Save the plot
            plt.figure(figsize=(12, 6))
            plt.plot(df[target_col], label='Observed')
            plt.plot(forecast_series, label='Forecast', color='red')
            plt.title('ARIMA Model Forecast')
            plt.xlabel('Date')
            plt.ylabel(target_col)
            plt.legend()
            plot_filename = 'forecast_plot.png'
            plot_filepath = os.path.join(settings.MEDIA_ROOT, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()

            # Pass the image URL to the template
            plot_url = os.path.join(settings.MEDIA_URL, plot_filename)
            return render(request, 'finmodapp/upload.html', {
                'form': form,
                'plot_url': plot_url
            })

    else:
        form = CSVUploadForm()

    return render(request, 'finmodapp/upload.html', {'form': form})






