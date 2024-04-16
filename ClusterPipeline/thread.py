import threading
from .models.ClusterProcessing import StockClusterGroup, StockClusterPeriod 
import traceback
import sys


class CreateGroupBackground(threading.Thread):
    def __init__(self, group_params, training_dict):
        self.group_params = group_params 
        self.group_params.initialize()
        self.training_dict = training_dict
        threading.Thread.__init__(self)


    def run(self):
        try: 
            print("Background thread started")
            self.create_new_group()
            print("Background thread finished")

        except Exception as e:
            print("Background thread failed")
            traceback.print_exc()
            self.group_params.delete()
            sys.exit(1)

    def create_new_group(self):
        cluster_group = StockClusterGroup.objects.create(group_params = self.group_params)
        cluster_group.generate_new_group(self.training_dict)




