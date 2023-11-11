from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMinMax
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
from copy import deepcopy
from collections import Counter

class SequenceElement: 
    def __init__(self,seq_x,seq_y,isTrain : bool, x_feature_dict, y_feature_dict,start_date, end_date,ticker) -> None:
        self.seq_x = seq_x
        self.seq_y = seq_y

        self.seq_x_scaled = deepcopy(seq_x)
        self.seq_y_scaled = deepcopy(seq_y)

        self.isTrain = isTrain
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.n_steps = len(seq_x)
        self.x_feature_dict = x_feature_dict
        self.y_feature_dict = y_feature_dict
        self.X_feature_sets = [] 
        self.y_feature_sets = []

    def create_array(sequence_elements,scaled = True):
        """
        Takes a list of sequence elements and returns a 3D array of the sequences
        """
        if scaled:
            X = np.array([sequence_element.seq_x_scaled for sequence_element in sequence_elements])
            y = np.array([sequence_element.seq_y_scaled for sequence_element in sequence_elements])
        else:
            X = np.array([sequence_element.seq_x for sequence_element in sequence_elements])
            y = np.array([sequence_element.seq_y for sequence_element in sequence_elements])
        return X,y



        
class ScalingMethod(Enum):
    """
    Enum class to encapsulate the different scaling methods available

    SBS - Sequence by sequence scaling. Each sequence is scaled independently
    SBSG - Sequence by sequence scale with grouped features. Each sequence is scaled independently but the same scaling is applied to all features
    QUANT_MINMAX - Quantile min max scaling. Each feature is scaled independently using the quantile min max scaling method
    UNSCALED - No scaling is applied
    """
    QUANT_MINMAX = 1
    SBS = 2
    SBSG = 3
    QUANT_MINMAX_G = 4
    UNSCALED = 5


class SequenceSet: 
    def __init__(self,group_params, training_dfs, test_dfs) -> None:

        self.group_params = group_params
        self.training_dfs = training_dfs
        self.test_dfs = test_dfs
    
    def set_n_steps(self,n_steps):
        self.n_steps = n_steps
    
    def scale_seq(self):
        pass

class StockSequenceSet(SequenceSet):
    def __init__(self,group_params,training_dfs,test_dfs) -> None:
        super().__init__(group_params=group_params,training_dfs=training_dfs,test_dfs=test_dfs)
    
    def create_combined_sequence(self):
        '''
        Creates the sequence for each dataframe and concatenates them together
        '''

        tickers = self.group_params.tickers
        n_steps = self.group_params.n_steps
        X_cols = self.group_params.X_cols
        y_cols = self.group_params.y_cols

        #empty 3D arrays
        train_seq_elements = []
        test_seq_elements = []

        for i in range(len(self.training_dfs)):
            training_set = self.training_dfs[i]
            test_set = self.test_dfs[i]

            ticker = tickers[i]
            train_elements, X_feature_dict, y_feature_dict = create_sequence(training_set, X_cols, y_cols, n_steps, ticker, isTrain = True)
            test_elements, X_feature_dict, y_feature_dict = create_sequence(test_set, X_cols, y_cols, n_steps, ticker, isTrain = False)

            
            train_seq_elements += train_elements
            test_seq_elements += test_elements
        
        # For QUANT_MIN_MAX features, the scaling occurs before the sequence is created, so we need to add the already scaled feature sets to the new sequence elements
        x_quant_min_max_feature_sets = [feature_set for feature_set in self.group_params.X_feature_sets if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value] 
        y_quant_min_max_feature_sets = [feature_set for feature_set in self.group_params.y_feature_sets if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value]  
        x_quant_min_max_feature_sets += [feature_set for feature_set in self.group_params.X_feature_sets if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value]
        y_quant_min_max_feature_sets += [feature_set for feature_set in self.group_params.y_feature_sets if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value]

        [seq_element.X_feature_sets.extend(x_quant_min_max_feature_sets) for seq_element in train_seq_elements]
        [seq_element.X_feature_sets.extend(x_quant_min_max_feature_sets) for seq_element in test_seq_elements]
        [seq_element.y_feature_sets.extend(y_quant_min_max_feature_sets) for seq_element in train_seq_elements]
        [seq_element.y_feature_sets.extend(y_quant_min_max_feature_sets) for seq_element in test_seq_elements]
        
        self.group_params.X_feature_dict = X_feature_dict
        self.group_params.y_feature_dict = y_feature_dict
        self.group_params.train_seq_elements = train_seq_elements
        self.group_params.test_seq_elements = test_seq_elements

        return train_seq_elements, test_seq_elements
    
    def scale_sequences(self): 
        scaler = SequenceScaler(self.group_params)
        train_seq_elements, test_seq_elements = scaler.scale_sequences() 
        self.group_params.train_seq_elements = train_seq_elements
        self.group_params.test_seq_elements = test_seq_elements
    
    def add_cuma_pctChg_features(self,feature_set): 
        '''
        Adds a cumulative percentage change feature to each sequence element in the sequence set

        This function creates a new feature that is the cumulative sum of the features from day 1 to t in time sequence.

        ie. If n_steps = 5, the cuma_pct_chg feature for a feature will be sum from day 1 to 5 of that feature. 
        '''

        example_seq_element = self.group_params.train_seq_elements[0]
        features_length = example_seq_element.seq_x_scaled.shape[1]

        X_feature_dict = self.group_params.X_feature_dict
        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        cols = feature_set.cols
        
        cuma_feature_set = deepcopy(feature_set)
        cuma_feature_set.name = "cuma_pctChg_vars"

        new_features = [] 
        for feature_name in cols: 
            idx = X_feature_dict[feature_name]
            new_feature_name = feature_name + "_cumulative"

            for seq in train_seq_elements:
                seq_x_scaled = seq.seq_x_scaled
                cumsum_pctChg = np.cumsum(seq_x_scaled[:,idx])
                seq_x_scaled = np.column_stack((seq_x_scaled,cumsum_pctChg))
                seq.seq_x_scaled = seq_x_scaled

            for seq in test_seq_elements:
                seq_x_scaled = seq.seq_x_scaled
                cumsum_pctChg = np.cumsum(seq_x_scaled[:,idx])
                seq_x_scaled = np.column_stack((seq_x_scaled,cumsum_pctChg))
                seq.seq_x_scaled = seq_x_scaled

            new_features.append(new_feature_name)

        cuma_feature_set.cols = []
        for new_feature in new_features:
            cuma_feature_set.cols.append(new_feature)
            X_feature_dict[new_feature] = features_length + new_features.index(new_feature) 
        
        self.group_params.X_feature_dict = X_feature_dict
        return cuma_feature_set
    
    def get_3d_array(self,scaled = True):
        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements
        X_train, y_train = SequenceElement.create_array(train_seq_elements,scaled)
        X_test, y_test = SequenceElement.create_array(test_seq_elements,scaled)
        return X_train, y_train, X_test, y_test
    
    def preprocess_pipeline(self,add_cuma_pctChg_features = False):
        self.create_combined_sequence()

        if self.group_params.train_seq_elements is None or self.group_params.test_seq_elements is None:
            raise ValueError("Sequence elements have not been created")
        if len(self.group_params.train_seq_elements) < 1 or len(self.group_params.test_seq_elements) < 1:
            raise ValueError("Sequence elements have not been created")
        if self.group_params.train_seq_elements[0].seq_x is None or self.group_params.test_seq_elements[0].seq_x is None:
            raise ValueError("Sequence elements have not been created")
        if self.group_params.train_seq_elements[0].seq_y is None or self.group_params.test_seq_elements[0].seq_y is None:
            raise ValueError("Sequence elements have not been created")
        if len(self.group_params.train_seq_elements[0].seq_x) < 1 or len(self.group_params.test_seq_elements[0].seq_x) < 1:
            raise ValueError("Sequence elements have not been created")
        if len(self.group_params.train_seq_elements[0].seq_y) < 1 or len(self.group_params.test_seq_elements[0].seq_y) < 1:
            raise ValueError("Sequence elements have not been created")

        self.scale_sequences()

        if self.group_params.train_seq_elements[0].seq_x_scaled is None or self.group_params.test_seq_elements[0].seq_x_scaled is None:
            raise ValueError("Sequence elements have not been created")
        if self.group_params.train_seq_elements[0].seq_y_scaled is None or self.group_params.test_seq_elements[0].seq_y_scaled is None:
            raise ValueError("Sequence elements have not been created")
        if len(self.group_params.train_seq_elements[0].seq_x_scaled) < 1 or len(self.group_params.test_seq_elements[0].seq_x_scaled) < 1:
            raise ValueError("Sequence elements have not been created")
        if len(self.group_params.train_seq_elements[0].seq_y_scaled) < 1 or len(self.group_params.test_seq_elements[0].seq_y_scaled) < 1:
            raise ValueError("Sequence elements have not been created")


        if add_cuma_pctChg_features:
            pct_Chg_feautures = (filter(lambda feature_set: feature_set.name == 'pctChg_vars' or feature_set.name == 'rolling_pctChg_vars', self.group_params.X_feature_sets))
            for pct_Chg_feauture in pct_Chg_feautures:
                cuma_feature_set = self.add_cuma_pctChg_features(pct_Chg_feauture)
                self.group_params.X_feature_sets.append(cuma_feature_set)



        

class SequenceScaler:
    """
    Class for dynamically scaling data based on the requirements of the features 
    Some scaling methods occur before sequences are created, others after.
    This method appropriately scales the data based on the requirements of the feature sets 
    and will return fully scaled data in 3D time sequence form. 
    """
    scalingMethods = ["SBS","SBSG","QUANT_MINMAX","UNSCALED"]

    def __init__(self,group_params) -> None:

        self.group_params = group_params
        
  
    def scale_sequences(self):
        '''
        Scales the sequences according to scaling method specified in the feature sets
        '''     
        X_feature_sets = self.group_params.X_feature_sets
        y_feature_sets = self.group_params.y_feature_sets
        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        #Then scale the SBS features in sequence form
        x_sbs_feature_sets = [feature_set for feature_set in X_feature_sets if feature_set.scaling_method.value == ScalingMethod.SBS.value]
        if len(x_sbs_feature_sets) > 0:
            train_seq_elements, test_seq_elements = self.scale_sbs(x_sbs_feature_sets,train_seq_elements,test_seq_elements)

        #Then scale the sbs features in sequence form
        x_sbsg_feature_sets = [feature_set for feature_set in X_feature_sets if feature_set.scaling_method.value == ScalingMethod.SBSG.value]
        if len(x_sbsg_feature_sets) > 0:
            train_seq_elements, test_seq_elements = self.scale_sbsg(x_sbsg_feature_sets,train_seq_elements,test_seq_elements)

        #Then scale the y features in sequence form
        y_sbs_feature_sets = [feature_set for feature_set in y_feature_sets if feature_set.scaling_method.value == ScalingMethod.SBS.value]
        if len(y_sbs_feature_sets) > 0:
            train_seq_elements, test_seq_elements = self.y_scale_sbs(y_sbs_feature_sets,train_seq_elements,test_seq_elements)

        return train_seq_elements, test_seq_elements
        
    
    def scale_sbs(self,feature_sets,train_seq_elements,test_seq_elements):
        """
        Scales the features in each feature_set sequence by sequence independant of each other 
        """
        train_seq_elements = deepcopy(train_seq_elements)
        test_seq_elements = deepcopy(test_seq_elements)
        X_feature_dict = self.group_params.X_feature_dict

        # Iterate through all the feature_sets requiring SBS scaling 
        for feature_set in feature_sets:
            # Extract the indices of the features in the feature set
            cols = feature_set.cols
            cols_indices = [X_feature_dict[col] for col in cols]

            # Iterate through each column, scaling each sequence in both the train and test set
            for index in cols_indices:
                for seq_element in train_seq_elements: 
                    ts = seq_element.seq_x
                    ts_scaled = seq_element.seq_x_scaled
                    scaler = MinMaxScaler(feature_set.range)
                    ts_scaled[:,index] = scaler.fit_transform(np.copy(ts[:,index].reshape(-1,1))).ravel()

                    feature_set_copy = deepcopy(feature_set)
                    feature_set_copy.scaler = scaler 
                    seq_element.X_feature_sets.append(feature_set_copy)

                for seq_element in test_seq_elements:
                    ts = seq_element.seq_x
                    ts_scaled = seq_element.seq_x_scaled
                    scaler = MinMaxScaler(feature_set.range)
                    ts_scaled[:,index] = scaler.fit_transform(np.copy(ts[:,index].reshape(-1,1))).ravel()
                    feature_set_copy = deepcopy(feature_set)
                    feature_set_copy.scaler = scaler 
                    seq_element.X_feature_sets.append(feature_set_copy)

        return train_seq_elements, test_seq_elements
    
    def scale_sbsg(self,feature_sets,train_seq_elements, test_seq_elements):
        """
        Scales the features in each feature_set sequence by sequence independant of each other 
        """
        train_seq_elements = deepcopy(train_seq_elements)
        test_seq_elements = deepcopy(test_seq_elements)
        X_feature_dict = self.group_params.X_feature_dict

        for feature_set in feature_sets:
            cols = feature_set.cols
            cols_indices = [X_feature_dict[col] for col in cols]
            
            # Create an instance of time series scaler
            # Iterate over each sequence and scale the specified features
            for i in range(len(train_seq_elements)):

                scaler = MinMaxScaler()
                # Extract the sequence
                ts = train_seq_elements[i].seq_x
                ts_scaled = train_seq_elements[i].seq_x_scaled
                
                # Vertically stack the columns to be scaled
                combined_series = ts[:, cols_indices].reshape(-1, 1)
                
                # Scale the combined series
                scaled_combined_series = scaler.fit_transform(np.copy(combined_series))
                # Split the scaled_combined_series back to the original shape and update the sequence
                ts_scaled[:, cols_indices] = scaled_combined_series.reshape(ts.shape[0], len(cols_indices))

    
                feature_set_copy = deepcopy(feature_set)
                feature_set_copy.scaler = scaler 
                train_seq_elements[i].X_feature_sets.append(feature_set_copy)
            
            
            for i in range(len(test_seq_elements)):
                scaler = MinMaxScaler()
                # Extract the sequence
                ts = test_seq_elements[i].seq_x
                ts_scaled = test_seq_elements[i].seq_x_scaled
                
                # Vertically stack the columns to be scaled
                combined_series = ts[:, cols_indices].reshape(-1, 1)
                # Scale the combined series
                scaled_combined_series = scaler.fit_transform(np.copy(combined_series))
                
                # Split the scaled_combined_series back to the original shape and update the sequence
                ts_scaled[:, cols_indices] = scaled_combined_series.reshape(ts.shape[0], len(cols_indices))
            
                feature_set_copy = deepcopy(feature_set)
                feature_set_copy.scaler = scaler 
                test_seq_elements[i].X_feature_sets.append(feature_set_copy)

    
        return train_seq_elements, test_seq_elements
    
    def y_scale_sbs(self,feature_sets,train_seq_elements,test_seq_elements): 
        """
        Scale the y features in each sequence with respect to each other. 
        This means if feature 1 is pctChg 1 day out and feature 2 is pctChg 5 days out, they will be on the same scale
        """
        train_seq_elements = deepcopy(train_seq_elements)
        test_seq_elements = deepcopy(test_seq_elements)
        y_feature_dict = self.group_params.y_feature_dict

        for feature_set in feature_sets:
            cols = feature_set.cols
            cols_indices = [y_feature_dict[col] for col in cols]

            for i,seq_ele in enumerate(train_seq_elements): 
                y_seq = seq_ele.seq_y

                scaler = MinMaxScaler(-1,1)
                scaled_y_seq = scaler.fit_transform(np.copy(y_seq[cols_indices].reshape(-1,1))).ravel()
                seq_ele.seq_y_scaled = scaled_y_seq
                feature_set_copy = deepcopy(feature_set)
                feature_set_copy.scaler = scaler
                seq_ele.y_feature_sets.append(feature_set_copy)

            for i,seq_ele in enumerate(test_seq_elements):
                y_seq = seq_ele.seq_y

                scaler = MinMaxScaler(-1,1)
                scaled_y_seq = scaler.fit_transform(np.copy(y_seq[cols_indices].reshape(-1,1))).ravel()
                seq_ele.seq_y_scaled = scaled_y_seq
                feature_set_copy = deepcopy(feature_set)
                feature_set_copy.scaler = scaler
                seq_ele.y_feature_sets.append(feature_set_copy)
        
        return train_seq_elements, test_seq_elements


def create_sequence(df, X_cols, y_cols, n_steps, ticker, isTrain):
    """
    Creates sequences of length n_steps from the dataframe df for the columns in X_cols and y_cols.
    """
    X_cols_list = list(X_cols)
    y_cols_list = sorted(list(y_cols)) ## Jenky work around to ensure that the columns for y target +1day,+2day etc are in the correct order
    df_cols = df[X_cols_list + y_cols_list]

    dates = df.index.tolist()
    sequence = df_cols.values
    
    # Get indices of X_cols and y_cols in the dataframe
    X_indices_df = [df_cols.columns.get_loc(col) for col in X_cols_list]
    y_indices_df = [df_cols.columns.get_loc(col) for col in y_cols_list]

    # Indices of features in the sequence (3D array)
    X_indices_seq = list(range(len(X_cols)))
    y_indices_seq = list(range(len(y_cols)))

    X_feature_dict = {col: index for col, index in zip(X_cols_list, X_indices_seq)}
    y_feature_dict = {col: index for col, index in zip(y_cols_list, y_indices_seq)}

    sequence_elements = []

    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence):
            break

        # Extract sequence for X
        seq_x = sequence[i:end_idx, :][:, X_indices_df]

        # Get sequence for y from the row at end_idx-1 of sequence for the columns in y_cols
        seq_y = sequence[end_idx - 1, y_indices_df]

        start_date = dates[i]
        end_date = dates[end_idx - 1]

        sequence_element = SequenceElement(seq_x, seq_y, isTrain, X_feature_dict, y_feature_dict, start_date, end_date,ticker)

        sequence_elements.append(sequence_element)
        
    return sequence_elements, X_feature_dict, y_feature_dict

