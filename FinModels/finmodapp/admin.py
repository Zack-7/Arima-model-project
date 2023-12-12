from django.contrib import admin
from .models import CSVFile
from import_export.admin import ImportExportModelAdmin
# Register your models here.
@admin.register (CSVFile)
class CSVAdmin (ImportExportModelAdmin):
    pass 