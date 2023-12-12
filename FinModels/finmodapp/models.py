from django.db import models
from django.db.models import JSONField

class CSVFile(models.Model):
    file = models.FileField(upload_to='csv_files/')
    content_json = JSONField(blank=True, null=True)

def __str__(self):
        return f"{self.file.name} - {self.pk}"

