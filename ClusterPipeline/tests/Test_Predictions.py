from django.test import TestCase
from ClusterPipeline.models import TSeriesPreproccesing as TSPP
from ClusterPipeline.models import Predictions as Pred
from ClusterPipeline.models import ClusterProcessing as CP
from datetime import date
from pandas.testing import assert_frame_equal
import copy

class TestPredictions(TestCase): 

    def setUp(self): 
        self.ticker = 'spy'
        self.start_date = '2015-01-01'
        # end date is today 
        self.end_date = date.today().strftime('%Y-%m-%d')
        self.interval = '1d'
        self.prediction = Pred.StockPredidction(ticker = self.ticker, interval = self.interval)
    
    def test_create_general_data_set(self):
        self.prediction.create_general_data_set(self.start_date, self.end_date)

        self.assertEqual(self.prediction.generic_dataset.group_params.tickers, [self.ticker])
        self.assertEqual(self.prediction.generic_dataset.group_params.interval, self.interval)
        self.assertEqual(self.prediction.generic_dataset.group_params.start_date, self.start_date)
        self.assertEqual(self.prediction.generic_dataset.group_params.end_date, self.end_date)

        df = self.prediction.generic_dataset.df 
        training_df = self.prediction.generic_dataset.training_df
        test_df = self.prediction.generic_dataset.test_df

        self.assertTrue(df is not None)
        self.assertTrue(training_df is not None)
        self.assertTrue(test_df is not None)

        self.assertTrue(len(df) > 0 )
        self.assertTrue(len(df) == len(test_df))
        self.assertTrue(len(training_df) == 0)

    def test_mirror_group(self):
        self.prediction.create_general_data_set(self.start_date, self.end_date) 
        
        target_features = ['sumPctChgclose+1','sumPctChgclose+2','sumPctChgclose+3','sumPctChgclose+4','sumPctChgclose+5','sumPctChgclose+6']

        # find a cluster group that matches the ticker 
        print(len(CP.StockClusterGroupParams.objects.all()))
        cluster_group = CP.StockClusterGroup.objects.get(id = 247)
        cluster_group_params = cluster_group.group_params

        print(cluster_group_params.tickers)
        print("loading saved group")
        cluster_group.load_saved_group()
        print("done loading saved group")

        y_feature_set = [feature_set for feature_set in cluster_group.group_params.y_feature_sets if feature_set.ticker == self.ticker][0]

        old_test_df = copy.deepcopy(self.prediction.generic_dataset.test_df)

        sequence_set = self.prediction.mirror_group(self.prediction.generic_dataset, cluster_group)

        new_test_df = copy.deepcopy(sequence_set.test_dfs[0])
        new_test_df[target_features] = y_feature_set.scaler.inverse_transform(new_test_df[target_features])

        cluster_group_params = sequence_set.group_params

        self.assertTrue(sequence_set is not None)
        self.assertTrue(len(cluster_group_params.train_seq_elements) == 0)   
        self.assertTrue(len(cluster_group_params.test_seq_elements)  > 0)
        self.assertTrue(len(cluster_group_params.future_seq_elements)  > 0)

        self.assertTrue(len(old_test_df) == len(new_test_df))

        # Work around for testing inverse transform after transforming is the same. If we quant scale the data, then the inverse transform will be slightly different 
        # for the values that are outside the quantitle
        # assert_frame_equal(old_test_df[target_features].head(1), new_test_df[target_features].head(1))

        for seq in cluster_group_params.future_seq_elements:
            print(seq)

    def test_predict_by_group(self): 
        self.prediction.create_general_data_set(self.start_date, self.end_date)

        # Extract cluster group that matches the situation (manually selected)

        cluster_group = CP.StockClusterGroup.objects.get(id = 308)
        cluster_group.load_saved_group()

        predictions = self.prediction.predict_by_group(cluster_group)
        
        pred_df = self.prediction.create_pred_df(predictions)
        print(pred_df)








 

