from django import forms
from .models import CSVFILE

class CSVUploadForm(forms.ModelForm):
    class meta:
        model = CSVFILE
        fields = ['file']
   
