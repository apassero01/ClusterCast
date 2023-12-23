from django.contrib import admin
from .models import ClusterProcessing as CP
from .models import SequencePreprocessing as SP
from .models import RNNModels as RM

admin.site.register(CP.StockClusterGroupParams)
admin.site.register(CP.StockClusterGroup)
admin.site.register(CP.StockCluster)
admin.site.register(CP.SupportedParams)
admin.site.register(RM.StepResult)
admin.site.register(RM.RNNModel)

# Register your models here.
