from django.contrib import admin
from .models import ClusterProcessing as CP
from .models import SequencePreprocessing as SP
from .models import RNNModels as RM
from .models import Predictions as Pred

admin.site.register(CP.StockClusterGroupParams)
admin.site.register(CP.StockClusterGroup)
admin.site.register(CP.StockCluster)
admin.site.register(CP.SupportedParams)
admin.site.register(RM.StepResult)
admin.site.register(RM.RNNModel)
admin.site.register(RM.ModelPrediction)
admin.site.register(Pred.StockPrediction)
admin.site.register(Pred.StockForcastTimeline)

# Register your models here.
