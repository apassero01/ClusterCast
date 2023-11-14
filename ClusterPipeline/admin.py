from django.contrib import admin
from .models import ClusterProcessing as CP

admin.site.register(CP.StockClusterGroupParams)
admin.site.register(CP.StockClusterGroup)
admin.site.register(CP.StockCluster)
admin.site.register(CP.SupportedParams)
admin.site.register(CP.StepResult)

# Register your models here.
