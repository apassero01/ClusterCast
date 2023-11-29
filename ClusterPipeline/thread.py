import threading
from .models.ClusterProcessing import StockClusterGroup, StockClusterPeriod 


class CreateGroupBackground(threading.Thread):
    def __init__(self, group_params):
        self.group_params = group_params 
        threading.Thread.__init__(self)

    def run(self):
        try: 
            print("Background thread started")
            self.create_new_group()

        except Exception as e:
            print("Background thread failed")
            print(e)
            self.group_params.delete()

    def create_new_group(self):
        cluster_group = StockClusterGroup.objects.create(group_params = self.group_params)
        cluster_group.generate_new_group(); 




