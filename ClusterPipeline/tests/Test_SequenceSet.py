import numpy as np
from copy import deepcopy
from django.test import TestCase
from ClusterPipeline.models import SequencePreprocessing as SP
from ClusterPipeline.models import ClusterProcessing as CP
from ClusterPipeline.models import TSeriesPreproccesing as TSPP
import pandas as pd
import random 

class TestStockSequenceSet(TestCase):

    def setUp(self) -> None:
        '''
        This method is called before each test
        '''
        tickers = ['AAPL', 'MSFT', 'AMD']
        start_date = '2010-01-01'
        target_cols = ['sumPctChgclose_1','sumPctChgclose_2','sumPctChgclose_3','sumPctChgclose_4','sumPctChgclose_5','sumPctChgclose_6']
        n_steps = 20
        interval = '1d'
        group_params = CP.StockClusterGroupParams(start_date = start_date, tickers = tickers, interval = interval, target_cols = target_cols, n_steps = n_steps)
        self.stockDataSet = TSPP.StockDataSet(group_params)
        
        #Create Dataset and do preprocessing 
        self.stockDataSet.preprocess_pipeline()

        #Data set creates a sequence set
        self.stockSequenceSet = self.stockDataSet.create_sequence_set()

    
    def test_create_sequence(self):
        '''
        Test that the non class function create_sequence works as expected

        Tests the following:
            - X_feature_dict contains key value pairs of the form {feature_name: feature_index}
                - To test this extract some data points from the dataframe and ensure that the same 
                value exists in the newly created sequenceElement in the index location indicated by the dict
                This Test must confirm that the feature names are correctly mapped to index location
            - y_target_dict contains key value pairs of the form {target_name: target_index} 
                - test to confirm that index locations of sumPctChgclose+1, +2... are in that order exactly in feature element
            - The sequence contains the correct value 
                - Test these by manually creating a few SequenceElements from the dataframe and compare the sequences to the returned sequences
        '''
        # df = self.stockSequenceSet.dfs[0]
        # X_cols = self.stockSequenceSet.X_cols
        # y_cols = self.stockSequenceSet.y_cols
        # n_steps = 10 # Whatever you would like 
        # ticker = self.stockSequenceSet.tickers[0]
        # sequence_elements, X_feature_dict, y_feature_dict = SP.create_sequence(df, X_cols, y_cols, n_steps, ticker)
        # TODO write tests above to ensure returned sequence elements are correct by comparing to values in the dataframe
        pass
    
    def test_create_comined_sequence(self):
        '''
        Test that the create_combined_sequence function works as expected

        Tests the following:
            - there are sequence_elements for every ticker in tickers list
            - the X_feature_dict and y_feature_dict contains feature name and index pairs for every feature in the sequence
            
        '''
        pass
    
    def test_rolling_pctChg_features(self):
        self.stockSequenceSet.create_combined_sequence()
        self.stockSequenceSet.scale_sequences()

        orig_group_params = deepcopy(self.stockSequenceSet.group_params)

        
        orig_train_array = SP.SequenceElement.create_array(np.copy(orig_group_params.train_seq_elements))[0]
        orig_test_array = SP.SequenceElement.create_array(np.copy(orig_group_params.test_seq_elements))[0]
        orig_feature_dict = deepcopy(orig_group_params.X_feature_dict)
        orig_num_features = len(orig_feature_dict.keys())
        #Create rolling pctChg features
        pct_Chg_feauture = next(filter(lambda feature_set: feature_set.name == 'pctChg_vars', orig_group_params.X_feature_sets))
        num_added_features = len(pct_Chg_feauture.cols)
        # print(pct_Chg_feauture.cols)

        self.stockSequenceSet.add_rolling_pctChg_features(deepcopy(pct_Chg_feauture))

        new_group_params = self.stockSequenceSet.group_params
        new_train_array = SP.SequenceElement.create_array(np.copy(new_group_params.train_seq_elements))[0]
        new_test_array = SP.SequenceElement.create_array(np.copy(new_group_params.test_seq_elements))[0]
        new_feature_dict = new_group_params.X_feature_dict
        new_num_features = len(new_feature_dict.keys())

        self.assertEqual(orig_train_array.shape[0], new_train_array.shape[0], 'train array shape[0] is not the same')
        self.assertEqual(orig_test_array.shape[0], new_test_array.shape[0],'test array shape[0] is not the same')
        self.assertEqual(orig_train_array.shape[1], new_train_array.shape[1],'train array shape[1] is not the same')
        self.assertEqual(orig_test_array.shape[1], new_test_array.shape[1],'test array shape[1] is not the same')
        

        self.assertEqual(orig_num_features + num_added_features, new_num_features, 'number features added is incorrect')
        self.assertEqual(new_num_features, new_train_array.shape[2],'new features not added to train array')
        self.assertEqual(new_num_features, new_test_array.shape[2],'new features not added to test array')

        self.assertTrue(np.array_equal(orig_train_array, new_train_array[:,:,0:orig_num_features]), 'original features not the same')
        self.assertTrue(np.array_equal(orig_test_array, new_test_array[:,:,0:orig_num_features]), 'original features not the same')

        for i in orig_feature_dict.keys():
            self.assertEqual(orig_feature_dict[i], new_feature_dict[i], 'feature dict not the same')
        
        # print(pct_Chg_feauture.cols)
        for col in pct_Chg_feauture.cols:
            new_name = col+'_cumulative'
            # print(new_name)

            self.assertTrue(new_name in new_feature_dict.keys(), 'new feature not added to feature dict')
            # print(new_feature_dict)
            for i in range(new_train_array.shape[0]):
                feature_row = new_train_array[i,:,new_feature_dict[col]]
                feature_row_cuma = new_train_array[i,:,new_feature_dict[new_name]]
                # print(feature_row)
                # print(feature_row_cuma)
                cumu_sum = 0
                for i in range(new_train_array.shape[1]):
                    cumu_sum += feature_row[i]
                    self.assertEqual(cumu_sum, feature_row_cuma[i], 'cumulative feature not calculated correctly')
        

    def test_scale_sbs(self):
        '''
        Test that the scale_sbs function works as expected
        Recall scaling sequence by sequence (SBS) means each sequenceElement is scaled independently of the others
        Example tests for this are in SequenceScalingTESTS.ipynb


        Tests the following:
            - All features with featureSet.scalingMethod == SBS are scaled between FeatureSet.range 
            - Ensure scaled data shape is the same as the original data shape
                - This can be done visually by looking at plotted points that should look the same but we want automated test
                - Should be able to do this by finding the correlation between the original and scaled data and that value should be -1 or 1 exactly. 
            - Each sequence element has a list of feature_sets and each feature set has a scaler saved to it. Extract the scaler with scaling_method = sbs and then  
            and inverse transform with the scaler and confirm it returns the original values in non scaled set (close to)
                - test this for a few random rows in the dataframe
        '''
        pass

    def test_scale_sbsg(self):
        '''
        Test that the scale_sbsg function works as expected
        Recall scaling sequence by sequence grouped (SBSG) means each sequenceElement is scaled independently of other sequences but features in the same sequence are scaled together
        For example if we have the 10day moving average and close price we want those features to be on the same scale but it is important that
        the distance between the two is preserved.

        Example tests for this are in SequenceScalingTESTS.ipynb

        Tests the following:
            - All features with featureSet.scalingMethod == SBSG are scaled between FeatureSet.range 
            - Ensure scaled data shape is the same as the original data shape for all features in feature sets 
                - This can be done visually by looking at plotted points that should look the same but we want automated test
                - Should be able to do this by finding the correlation between the original and scaled data and that value should be -1 or 1 exactly. 
                - might have to check correlation for each feature in feature set individually 
            - Each sequence element has a list of feature_sets and each feature set has a scaler saved to it. Extract the scaler with scaling_method = sbsg and then  
            and inverse transform with the scaler and confirm it returns the original values in non scaled set (close to)
                - test this for a few random rows in the dataframe
        '''
        pass

    def test_y_scale_sbs(self):
        '''
        Test that the y_scale_sbs function works as expected. For a 1D array of y vales in each sequence element, scale each sequence element independently of the others

        Tests the following: 
            - All y values in each sequence element are scaled between -1 and 1
            - Each sequenceElement.y_seq should contain a value of -1 and 1 exactly
            - Ensure scaled data shape is the same as the original data 
                - again use correlation to test this
            - Each sequence element has a scaler saved to it. Extract the scaler and then inverse transform to confirm
            we get the original values matching unscaled seq elements
                - test this for a few random rows in the dataframe
        '''
        pass

    def test_scale_sequences(self):
        '''
        Test that the scale_sequence function works as expected
        This function inside the StockSequenceSet class calls the same method in the SequenceScaler class
        Test both of these methods inside of this test. 
        
        Tests the following: 
            - All values in each sequence element are scaled between -1 and 1
            - Each sequenceElement has a list of feature sets the same length as X_feature_sets
                - there should also be the same number of QUANT_MINMAX, SBS and SBSG feature sets in the scaler.X_feature_sets list 
            - The length of each sequenceElement.X_seq and sequenceElement.y_seq is the same as the length
        '''
        pass
    
        

    


    

        