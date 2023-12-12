from django.test import TestCase, Client
from django.urls import reverse
from .models import CSVFile
import os
import pandas as pd
from .arima_utils import load_and_preprocess_data, build_and_fit_arima_model

class FinmodAppTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_upload_file_view(self):
        # Path to the existing CSV file
        file_path = 'C:\\Users\\Onke Gwala\\dailydeli.csv'

        with open(file_path, 'rb') as file:
            response = self.client.post(reverse('upload_file'), {'file': file}, format='multipart')

        self.assertEqual(response.status_code, 302)  # Check for redirect
        self.assertEqual(CSVFile.objects.count(), 1)  # Check if file is saved

class UtilsTests(TestCase):
    def test_load_and_preprocess_data(self):
        # Path to the existing CSV file
        file_path = 'C:\\Users\\Onke Gwala\\dailydeli.csv'

        df = pd.read_csv(file_path)
        processed_df = load_and_preprocess_data(df, 'date', 'meantemp')
        self.assertIsNotNone(processed_df)

    def test_build_and_fit_arima_model(self):
        file_path = 'C:\\Users\\Onke Gwala\\dailydeli.csv'
        df = pd.read_csv(file_path)  # Use the actual data
        model_fit = build_and_fit_arima_model(df, 'meantemp', (1, 1, 1))
        self.assertIsNotNone(model_fit)