from datetime import datetime
from .SequencePreprocessing import StockSequenceSet, SequenceElement, ScalingMethod
from .TSeriesPreproccesing import StockDataSet
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import plotly.graph_objects as go
import math
from keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU,BatchNormalization, Input, Concatenate, Attention, Masking
from tensorflow.keras.optimizers import Adam
import pandas as pd
import tensorflow as tf
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pickle
from django.db import models
from tensorflow.keras.backend import clear_session
import os
from tensorflow.keras.callbacks import EarlyStopping
import sys
from .RNNModels import RNNModel, ModelTypes
from tensorflow.keras.initializers import glorot_uniform, zeros


class SupportedParams(models.Model):
    """
    Class to contain potential parameters for running the pipeline
    We can run methods in this class to add parameters to the database that can be used for front end forms 
    """
    features = models.JSONField(default=list)
    name = models.CharField(max_length=100,default="")

    pct_chg_features = models.JSONField(default=list)
    rolling_features = models.JSONField(default=list)
    trend_features = models.JSONField(default=list)
    price_features = models.JSONField(default=list)
    cuma_features = models.JSONField(default=list)

    def generate_features(self):
        '''
        Method to generate a list of all the features that the current pipeline produces
        '''
        tickers = ['spy']
        start = '2020-01-01'
        target_cols = ['sumpctChgclose_1','sumpctChgclose_2','sumpctChgclose_3','sumpctChgclose_4','sumpctChgclose_5','sumpctChgclose_6']
        n_steps = 2
        interval = '1d'
        cluster_features = ['pctChgclose_cumulative']
        group_params = StockClusterGroupParams(start_date = start, tickers = tickers, interval = interval, target_cols = target_cols, n_steps = n_steps,cluster_features = cluster_features)
        group_params.scaling_dict = {
                    'price_vars': ScalingMethod.SBSG,
                    'trend_vars' : ScalingMethod.SBS,
                    'pctChg_vars' : ScalingMethod.QUANT_MINMAX,
                    'rolling_vars' : ScalingMethod.QUANT_MINMAX_G,
                    'target_vars' : ScalingMethod.UNSCALED
                    }
        cluster_group = StockClusterGroup()
        cluster_group.set_group_params(group_params)
        cluster_group.create_data_set()
        cluster_group.create_sequence_set()

        self.features = list(cluster_group.sequence_set.group_params.X_cols)
    
    def generate_features_by_type(self):
        '''
        Similiar types of features are grouped together. This method generates a list of features for each type
        '''
        tickers = ['spy']
        start = '2020-01-01'
        target_cols = ['sumpctChgclose_1','sumpctChgclose_2','sumpctChgclose_3','sumpctChgclose_4','sumpctChgclose_5','sumpctChgclose_6']
        n_steps = 2
        interval = '1d'
        cluster_features = ['pctChgclose_cumulative']
        group_params = StockClusterGroupParams(start_date = start, tickers = tickers, interval = interval, target_cols = target_cols, n_steps = n_steps,cluster_features = cluster_features)
        group_params.scaling_dict = {
                    'price_vars': ScalingMethod.SBSG,
                    'trend_vars' : ScalingMethod.SBS,
                    'pctChg_vars' : ScalingMethod.QUANT_MINMAX,
                    'rolling_vars' : ScalingMethod.QUANT_MINMAX_G,
                    'target_vars' : ScalingMethod.UNSCALED
                    }
        cluster_group = StockClusterGroup()
        cluster_group.set_group_params(group_params)
        cluster_group.create_data_set()
        cluster_group.create_sequence_set()

        self.pct_chg_features = next(filter(lambda feature_set: feature_set.name == 'pctChg_vars', cluster_group.group_params.X_feature_sets)).cols
        self.cuma_features = next((filter(lambda feature_set: "cum" in feature_set.name, cluster_group.group_params.X_feature_sets))).cols
        self.price_features = next((filter(lambda feature_set: "price" in feature_set.name, cluster_group.group_params.X_feature_sets))).cols
        self.trend_features = next((filter(lambda feature_set: "trend" in feature_set.name, cluster_group.group_params.X_feature_sets))).cols

        rolling_feat= list((filter(lambda feature_set: "rolling" in feature_set.name, cluster_group.group_params.X_feature_sets)))
        for feature in rolling_feat: 
            self.rolling_features += feature.cols

class ClusterGroupParams(models.Model): 
    '''
    Class to contain parameters for running the pipeline. This class is abstract
    '''
    start_date = models.DateField()
    end_date = models.DateField()
    n_steps = models.IntegerField()
    scaling_dict = models.JSONField(default=dict)
    name = models.CharField(max_length=100,default="")
    X_feature_dict = models.JSONField(default=dict)
    y_feature_dict = models.JSONField(default=dict)

    class Meta:
        abstract = True
    
    def initialize(self):
        self.X_cols = None
        self.y_cols = None
        self.data_sets = [] 
        self.train_seq_elements = None
        self.test_seq_elements = None
        self.name = str(self.tickers) + str(self.start_date) + "-" + str(self.end_date) + str(self.n_steps) +"steps"
    
    def set_scaling_dict(self,scaling_dict):
        self.scaling_dict = {
            key: value.value for key, value in scaling_dict.items()
        }
    def get_scaling_dict(self):
        return {
            key: ScalingMethod(value) for key, value in self.scaling_dict.items()
        }

class StockClusterGroupParams(ClusterGroupParams): 
    tickers = models.JSONField(default=list)
    target_cols = models.JSONField(default=list)
    interval = models.CharField(max_length=10)
    cluster_features = models.JSONField(default=list)
    training_features = models.JSONField(default=list)

    


class ClusterPeriod: 
    def __init__(self,start_date,end_date): 
        self.start_date = start_date
        self.end_date = end_date
        self.current_cluster_group = None
    
    def create_cluster_group(self,group_params): 
        pass

class StockClusterPeriod(ClusterPeriod):
    
    def __init__(self,start_date,end_date):
        super().__init__(start_date,end_date)

    
    def create_cluster_params(self, tickers, target_cols, n_steps, cluster_features, interval = '1d'):
        return StockClusterGroupParams(self.start_date,tickers,target_cols,n_steps,cluster_features,interval,self.end_date)
    
    def update_param_scaling_dict(self, group_params, scaling_dict):
        group_params.scaling_dict = scaling_dict

    def create_cluster_group(self, group_params):
        cluster_group = StockClusterGroup(group_params)
        cluster_group.create_data_set()
        cluster_group.create_sequence_set()

        self.current_cluster_group = cluster_group
    
    def cluster_current_group(self,alg = 'TSKM',metric = "euclidean"):
        if self.cluster_current_group is None:
            raise ValueError("No cluster group has been created yet")
        
        self.current_cluster_group.run_clustering(alg,metric)
        self.current_cluster_group.create_clusters()
    
    def display_current_group(self):
        if self.current_cluster_group is None:
            raise ValueError("No cluster group has been created yet")
        self.current_cluster_group.display_all_clusters()
    



class ClusterGroup(models.Model): 
    '''
    Class to encapsulate a group of clusters ie. The class that can run a clustering algorithm on all the data points, and 
    then create cluster objects for each label returned from the clustering algorithm. This class is abstract. 
    '''
    n_clusters = models.IntegerField(default=0)
    train_labels = models.JSONField(default=list)
    test_labels = models.JSONField(default=list)

    class Meta:
        abstract = True
        

    def set_group_params(self,group_params): 
        self.group_params = group_params

    def create_data_set(self): 
        pass

    def create_sequence_set(self):
        pass

class StockClusterGroup(ClusterGroup):
    '''
    Implementation of the ClusterGroup class for stock data
    '''
    group_params = models.OneToOneField(StockClusterGroupParams, on_delete=models.CASCADE, related_name='group_params')

    def create_data_set(self):
        '''
        Method to create a StockDataSet object from the group_params
        '''
        self.data_sets = [] 
        for ticker in self.group_params.tickers:
            self.data_set = StockDataSet(self.group_params, ticker)
            self.data_set.preprocess_pipeline()
            self.data_sets.append(self.data_set)
            self.group_params = self.data_set.group_params
        
        self.group_params.data_sets = self.data_sets

    
    def create_sequence_set(self):
        '''
        Method to create a StockSequenceSet object from the data_set
        '''

        self.sequence_set = StockSequenceSet(self.group_params)
        self.sequence_set.preprocess_pipeline(add_cuma_pctChg_features=True)
        self.group_params = self.sequence_set.group_params
    
    def run_clustering(self,alg = 'TSKM',metric = "euclidean"):
        '''
        Method to run a clustering algorithm on the data set. 

        Parameters:
        alg: The clustering algorithm to use. Currently only supports TimeSeriesKMeans
        metric: The metric to use for the clustering algorithm. default is euclidean
        '''
        self.get_3d_array()
        X_train_cluster = self.filter_by_features(self.X_train, self.group_params.cluster_features)
        X_test_cluster = self.filter_by_features(self.X_test, self.group_params.cluster_features)


        if alg == 'TSKM':
            # n_clusters = self.determine_n_clusters(X_train_cluster,metric)
            n_clusters = math.ceil(math.sqrt(len(X_train_cluster))) // 5
            # n_clusters = 1
            self.cluster_alg = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,random_state=3)
        
        self.train_labels = list(self.cluster_alg.fit_predict(X_train_cluster))
        self.test_labels = list(self.cluster_alg.predict(X_test_cluster))

        self.train_labels = [int(x) for x in self.train_labels]
        self.test_labels = [int(x) for x in self.test_labels]

        self.cluster_distances = self.cluster_alg.transform(X_train_cluster)
        self.cluster_centers = self.cluster_alg.cluster_centers_

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        if len(self.train_labels) != len(train_seq_elements):
            raise ValueError("The number of labels does not match the number of sequences")
        if len(self.test_labels) != len(test_seq_elements):
            raise ValueError("The number of labels does not match the number of sequences")

        # np.random.shuffle(self.train_labels)
        # np.random.shuffle(self.test_labels)

        for i in range(len(train_seq_elements)):
            seq_element = train_seq_elements[i]
            seq_element.cluster_label = self.train_labels[i]
        for i in range(len(test_seq_elements)):
            seq_element = test_seq_elements[i]
            seq_element.cluster_label = self.test_labels[i]
    
    def determine_n_clusters(self,X_train_cluster,metric = "euclidean"):
        '''
        Method that utilizes the elbow method to determine the optimal number of clusters
        '''
        min_clusters = math.ceil(math.sqrt(len(X_train_cluster))) // 8
        max_clusters = math.ceil(math.sqrt(len(X_train_cluster))) // 2 

        wcss = []
        self.K = range(min_clusters,max_clusters,4)
        for k in self.K:
            cluster_alg = TimeSeriesKMeans(n_clusters=k, metric=metric,random_state=3)
            train_labels = cluster_alg.fit_predict(X_train_cluster)
            wcss.append(cluster_alg.inertia_)
        
        kn = KneeLocator(self.K,wcss, curve='convex', direction='decreasing')

        self.wcss = wcss
        return kn.knee


    
    def create_clusters(self):
        '''
        Method to create a StockCluster object for each label returned from the clustering algorithm
        '''

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        train_labels_unique = np.unique([self.train_labels])

        self.clusters = []

        for label in train_labels_unique:
            # Get all the sequences that belong to this cluster
            cur_train_seq_elements = [x for x in train_seq_elements if x.cluster_label == label]
            cur_test_seq_elements = [x for x in test_seq_elements if x.cluster_label == label]

            # Get the distances of all the sequences in this cluster to the cluster center
            cluster_distances = self.cluster_distances[self.train_labels == label, label]

            std_dev = np.std(cluster_distances)

            iqr = np.percentile(cluster_distances, 75) - np.percentile(cluster_distances, 25)

            metrics = {"std_dev": std_dev, "iqr": iqr}

            if len(cur_train_seq_elements) == 0 or len(cur_test_seq_elements) == 0:
                continue

            # Create the object and pass in the label, cluster_group and associated metrics 
            cluster = StockCluster.objects.create(label=label, cluster_group=self, cluster_metrics=metrics)
            cluster.initialize(cur_train_seq_elements,cur_test_seq_elements)
            self.clusters.append(cluster)



    
    def train_all_rnns(self,model_features, fine_tune = False):
        '''
        Method to train an RNN for each cluster. The functionality for training an RNN is encapsulated in the train_rnn method of the StockCluster class.
        This method iterates over the clusters and trains the model. 

        Parameters:
        model_features: The features to use for training the RNN.
        fine_tune: Boolean to indicate whether we are training a base model and fine tunining it on the individual clusters. 
        '''
        # model_features = ['pctChgclose_cumulative','pctChgvolume_cumulative']
        # model_features = ['pctChgclose', 'pctChgvolume','sumpctChgclose_6','sumpctChgvolume_6','sumpctChgema50_1','sumpctChgema10_1','sumpctChgema50_6','sumpctChgema10_6']
        model = None 
        # print(model_features)
        if fine_tune:
            self.train_general_model(model_features)
            model = self.general_model
            
        
        self.filtered_clusters = []

        for cluster in self.clusters:
            cluster.train_rnn(model_features,model)
            cluster.filter_results()
            if cluster.num_results > 0:
                self.filtered_clusters.append(cluster)
                cluster.serialize()
                cluster.save()
                del cluster.model
                cluster.model = None
            else: 
                cluster.delete()

        self.group_params.save() 

    
    def train_general_model(self,model_features):
        '''
        Method for training a general model on the entire data set. This model will be used for fine tuning on the individual clusters.
        '''
        X_train = self.filter_by_features(self.X_train, model_features)
        X_test = self.filter_by_features(self.X_test, model_features)
        y_train = self.y_train
        y_test = self.y_test

        self.general_model = create_model(len(model_features))
        
        self.general_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)
        clear_session()


    def filter_by_features(self,seq, feature_list):
        '''
        Method to filter a 3d array of sequences by a list of features.
        '''
        seq = seq.copy()
        indices = [self.group_params.X_feature_dict[x] for x in feature_list]
        # Using numpy's advanced indexing to select the required features
        return seq[:, :, indices]
    
    def get_3d_array(self): 
        '''
        Method to get the 3d array of sequences from the sequence set
        '''
        self.X_train, self.y_train, self.X_test, self.y_test = self.sequence_set.get_3d_array()
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def load_saved_clusters(self):
        '''
        Method to load saved clusters from the database
        '''
        self.clusters = StockCluster.objects.filter(cluster_group = self)

    def generate_new_group(self):
        '''
        Method to generate a new group. This method is used when the user wants to run the pipeline from scratch
        '''
        model_features = self.group_params.training_features;
        self.create_data_set()
        self.create_sequence_set()
        self.run_clustering()
        self.create_clusters()
        self.train_all_rnns(model_features=model_features)

        self.save()
    
    def load_saved_group(self):
        '''
        Method to load a saved group from the database. We perform the preprocessing steps that are necessary and avoid the ones that are not
        '''
        self.create_data_set()
        self.create_sequence_set()
        self.load_saved_clusters()

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        for i in range(len(train_seq_elements)):
            seq_element = train_seq_elements[i]
            seq_element.cluster_label = self.train_labels[i]
        for i in range(len(test_seq_elements)):
            seq_element = test_seq_elements[i]
            seq_element.cluster_label = self.test_labels[i]


        self.clusters = StockCluster.objects.filter(cluster_group = self)

        for cluster in self.clusters:
            cluster_label = cluster.label
            cur_train_seq_elements = [x for x in train_seq_elements if x.cluster_label == cluster_label]
            cur_test_seq_elements = [x for x in test_seq_elements if x.cluster_label == cluster_label]
            cluster.initialize(cur_train_seq_elements,cur_test_seq_elements)

        




class Cluster(models.Model):
    '''
    Abstract class to encapsulate a cluster. This class is abstract. A cluster is a group of sequences that were assigned the same label in 
    a clustering algorithm. 

    label: is the label assigned to this cluster
    model_file_string: is the path to the directory where the model is saved
    cluster_metrics: is a dictionary containing metrics for this cluster
    '''
    label = models.IntegerField(default=-1)
    elements_file_string = models.CharField(max_length=100,default="")
    model_file_string = models.CharField(max_length=100,default="")
    cluster_metrics = models.JSONField(default=dict)
    class Meta:
        abstract = True
    
    def initialize(self,train_seq_elements,test_seq_elements):
        '''
        Method to initialize the cluster. This method is called after the cluster is created and the sequences are assigned to it.
        '''
        self.train_seq_elements = train_seq_elements
        self.test_seq_elements = test_seq_elements
        self.get_3d_array()

    def get_3d_array(self):
        '''
        Method to get the 3d array of sequences from the sequence set
        '''
        if len(self.train_seq_elements) == 0 or len(self.test_seq_elements) == 0:
            raise ValueError("No sequences in this cluster")
        
        self.X_train, self.y_train = SequenceElement.create_array(self.train_seq_elements)
        self.X_test, self.y_test = SequenceElement.create_array(self.test_seq_elements)

        return self.X_train, self.y_train, self.X_test, self.y_test
    

    def serialize(self):
        '''
        Method to serialize the cluster. This method saves the model and the sequences to the database
        '''
        group_name = self.cluster_group.group_params.name
    
        self.model_file_string = f"SavedModels/{group_name}/Cluster{self.label}/"


        # Check if the directory exists
        if not os.path.exists(self.model_file_string):
            # Create the directory if it doesn't exist
            print("Creating directory " + self.model_file_string)
            os.makedirs(self.model_file_string)

        print("Saving model to " + self.model_file_string)  
        with open(self.model_file_string+"Model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
    
    
    def deserialize_model(self):
        '''
        Method to load the model from the database
        '''
        with open(self.model_file_string, 'rb') as f:
            self.model = pickle.load(f)

class StockCluster(Cluster):
    '''
    Implementation of the Cluster class for stock data
    '''
    cluster_group = models.ForeignKey(StockClusterGroup, on_delete=models.CASCADE, related_name='clusters_obj')
    
    def remove_outliers(self):
        pass

    def visualize_cluster(self, isTrain = True, y_range = [-1,1]):

        if isTrain:
            arr_3d = self.X_train
        else:
            arr_3d = self.X_test

        X_cluster = self.cluster_group.filter_by_features(arr_3d, self.cluster_group.group_params.cluster_features)

        traces = [] 
        avg_cluster = np.mean(X_cluster,axis = 0)
        std_cluster = np.std(X_cluster,axis = 0)

        upper_bound = avg_cluster + std_cluster
        lower_bound = avg_cluster - std_cluster
        
        x = np.arange(avg_cluster.shape[0])

        for feature_idx in range(avg_cluster.shape[1]):
            y_avg = np.ones(avg_cluster.shape[0]) * feature_idx
            z_avg = avg_cluster[:, feature_idx]
            z_upper = upper_bound[:, feature_idx]
            z_lower = lower_bound[:, feature_idx]
            traces.append(go.Scatter3d(x=x, y=y_avg, z=z_avg, mode='lines', line=dict(color='red', width=2)))
            traces.append(go.Scatter3d(x=x, y=y_avg, z=z_upper, mode='lines', line=dict(color='green', width=1)))
            traces.append(go.Scatter3d(x=x, y=y_avg, z=z_lower, mode='lines', line=dict(color='blue', width=1)))

        fig = go.Figure(data=traces)

        fig.update_layout(title="Cluster " + str(self.label),
                        scene=dict(xaxis_title='Time',
                                    yaxis_title='Feature Index',
                                    zaxis_title='Value',
                                    zaxis = dict(range=y_range)))

        return fig
    
    def visualize_future_distribution(self, isTest = True):
        '''
        Create stacked box and whisker plots for the predicted and real values
        '''

        fig = go.Figure()
        step_results =  StepResult.objects.filter(cluster = self) 

        for step_result in step_results:
            i = step_result.steps_in_future 

            if isTest:
                fig.add_trace(go.Box(y=step_result.predicted_values, name=f'Predicted {i}')) 
                fig.add_trace(go.Box(y=step_result.actual_values, name=f'Real {i}'))
            else:
                fig.add_trace(go.Box(y=self.y_train[:,i], name=f'Predicted {i}'))

        fig.update_layout(
            title='Future Performance of Cluster',
            xaxis_title='Steps in future',
            yaxis_title='Cumulative Percent Change'
        ) 

        return fig 

    
    def train_rnn(self,model_features,general_model = None):
        if len(self.X_train) == 0 or len(self.X_test) == 0:
            return
        
        model_features = ['pctChgclose', 'pctChgvolume','sumpctChgclose_6','sumpctChgvolume_6','sumpctChgema50_1','sumpctChgema10_1','sumpctChgema50_6','sumpctChgema10_6']

        X_train_filtered = self.cluster_group.filter_by_features(self.X_train, model_features)
        X_test_filtered = self.cluster_group.filter_by_features(self.X_test, model_features)

        non_float32_elements = X_train_filtered[np.where(X_train_filtered.dtype != np.float32)]

        if non_float32_elements.size == 0:
            print("There are no non-float32 elements in the array.")
        else:
            print("There are non-float32" + str(non_float32_elements.size) + "elements in the array." + " Out of "+ str(X_train_filtered.size) + " elements.")

        y_train = self.y_train
        y_test = self.y_test

        if general_model is None: 
            self.model = create_model(len(model_features))
        else: 
            freeze_up_to_layer_name = 'encoder_lstm_2' # was repeat vector
            self.model = clone_for_tuning(general_model, freeze_up_to_layer_name, learning_rate=0.0001)


        # self.model = create_attention(len(model_features),6,1)
        self.model = RNNModel(ModelTypes.AE, input_shape = X_train_filtered.shape, output_shape = y_train.shape[1],num_encoder_layers=2)
        self.model.addGRULayer(50,return_sequences=True,activation='tanh')
        self.model.addDropoutLayer(0.2)
        self.model.addGRULayer(30,return_sequences=True,activation='tanh')
        # self.model.addGRULayer(64,return_sequences=True)
        self.model.addDropoutLayer(0.2)
        self.model.addGRULayer(20,return_sequences=True,activation='tanh')
        self.model.addGRULayer(15,return_sequences=True,activation='tanh')
        # self.model.addLSTMLayer(32,return_sequences=True)
        # self.model.addLSTMLayer(6,return_sequences=True)
        self.model.buildModel()
        # early_stopping = EarlyStopping(
        #     monitor='val_loss',  # Metric to monitor (e.g., validation loss)
        #     patience=15,          # Number of epochs with no improvement before stopping
        #     restore_best_weights=True  # Restore model weights to the best epoch
        # )

        # Train the model with early stopping

        self.model.fit(X_train_filtered, y_train, epochs=500, batch_size=32, validation_data=(X_test_filtered, y_test))

        predicted_y = self.model.predict(X_test_filtered)
        print(predicted_y.shape)
        
        predicted_y = np.squeeze(predicted_y, axis=-1)

        num_days = predicted_y.shape[1]  # Assuming this is the number of days
        results = pd.DataFrame(predicted_y, columns=[f'predicted_{i+1}' for i in range(num_days)])

        
        for i in range(num_days):
            results[f'real_{i+1}'] = y_test[:, i]

        # Calculate the P/L for each predictio
        # Generate output string with accuracies
        self.step_results = [] 
        for i in range(num_days):
            step_result = StepResult.objects.create(steps_in_future = i+1, cluster = self,train_set_length = len(X_train_filtered), test_set_length = len(y_test))
            same_day = ((results[f'predicted_{i+1}'] > 0) & (results[f'real_{i+1}'] > 0)) | \
                    ((results[f'predicted_{i+1}'] < 0) & (results[f'real_{i+1}'] < 0))
            accuracy = round(same_day.mean() * 100,2)
            w_accuracy = round(weighted_dir_acc(results[f'predicted_{i+1}'], results[f'real_{i+1}']),2)
            p_l = profit_loss(results[f'predicted_{i+1}'], results[f'real_{i+1}'])


            step_result.predicted_values = list(results[f'predicted_{i+1}'])
            step_result.actual_values = list(results[f'real_{i+1}'])



            with open("test_output.txt", "a") as f:
                # Writing headers
                f.write("Predicted, Real\n")
                
                # Writing each row's predicted and real values
                for _, row in results.iterrows():
                    f.write(f"{row[f'predicted_{i+1}']}, {row[f'real_{i+1}']}\n")
                
                # Writing additional information
                f.write("\n")  # New line for separation
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"Weighted Accuracy: {w_accuracy}\n")
                f.write(f"Profit/Loss: {p_l}\n")

            step_result.dir_accuracy = accuracy
            step_result.p_l = p_l
            step_result.weighted_dir_acc = w_accuracy
            step_result.predicted_return = round(results[f'predicted_{i+1}'].mean(),2)
            step_result.actual_return = round(results[f'real_{i+1}'].mean(),2)
            step_result.save()

        clear_session()
    
    def filter_results(self,threshhold = 0.2,test_set_length = 30):
        self.step_results = StepResult.objects.filter(cluster = self)
        for result in self.step_results:
            if result.dir_accuracy < threshhold or result.test_set_length < test_set_length:
                result.delete()
        self.num_results = len(self.step_results)
    
    def generate_results(self):
        results = {
            "train_set_length": len(self.X_train),
            "test_set_length": len(self.y_test),
            "cluster_label": int(self.label),
            "step_accuracy": [],
            "step_accuracy_weighted": [],
            "step_predicted_return": [],
            "step_actual_return": [],
            "step_p_l": [],
        }
        self.step_results = self.cluster_results.all()
        for result in self.step_results:
            results["step_accuracy"].append(int(result.dir_accuracy))
            results["step_accuracy_weighted"].append(int(result.weighted_dir_acc))
            results["step_predicted_return"].append(float(result.predicted_return))
            results["step_actual_return"].append(float(result.actual_return))
            results["step_p_l"].append(float(result.p_l))
        return results
    






def weighted_dir_acc(predicted, actual):
    directional_accuracy = (np.sign(predicted) == np.sign(actual)).astype(int)
    magnitude_difference = np.abs(np.abs(predicted) - np.abs(actual)) + 1e-6
    weights = np.abs(actual) / magnitude_difference
    return np.sum(directional_accuracy * weights) / np.sum(weights) * 100

def profit_loss(predicted, actual):
    p_l = 0
    for i in range(len(predicted)):
        if predicted[i] > 0:
            if actual[i] > 0:
                p_l += abs(actual[i])
            else:
                p_l -= abs(actual[i])
        else:
            if actual[i] < 0:
                p_l += abs(actual[i])
            else:
                p_l -= abs(actual[i])
    return p_l/len(predicted)

def create_model(input_shape):
    model_lstm = Sequential()
    
    # model_lstm.add(Masking(mask_value=0.0, input_shape=(None, input_shape), name='masking_layer'))
    # Encoder
    model_lstm.add(LSTM(units=100, activation='tanh', return_sequences=True,input_shape=(None, input_shape), name='encoder_lstm_1'))
    model_lstm.add(BatchNormalization(name='encoder_bn_1'))
    model_lstm.add(Dropout(0.2, name='encoder_dropout_1'))

    model_lstm.add(LSTM(units=100, activation='tanh', return_sequences=True, name='encoder_lstm_2'))
    model_lstm.add(Dropout(0.2, name='encoder_dropout_2'))

    model_lstm.add(LSTM(units=50, activation='tanh', return_sequences=True, name='encoder_lstm_3'))
    model_lstm.add(Dropout(0.2, name='encoder_dropout_3'))

    model_lstm.add(LSTM(units=25, activation='tanh', name='encoder_lstm_4'))
    model_lstm.add(BatchNormalization(name='encoder_bn_4'))
    model_lstm.add(Dropout(0.2, name='encoder_dropout_4'))
    
    # Repeat Vector
    model_lstm.add(RepeatVector(6, name='repeat_vector'))  # Assuming you are predicting for 6 steps

    # Decoder
    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True, name='decoder_lstm_1'))
    model_lstm.add(BatchNormalization(name='decoder_bn_1'))
    model_lstm.add(Dropout(0.2, name='decoder_dropout_1'))

    model_lstm.add(LSTM(units=100, activation='tanh', return_sequences=True, name='decoder_lstm_2'))
    model_lstm.add(BatchNormalization(name='decoder_bn_2'))
    model_lstm.add(Dropout(0.2, name='decoder_dropout_2'))
    
    model_lstm.add(TimeDistributed(Dense(1), name='time_distributed_output'))

    # Compile
    optimizer = Adam(learning_rate=0.001)
    model_lstm.compile(optimizer=optimizer, loss="mae")

    return model_lstm


def clone_for_tuning(base_model, freeze_up_to_layer_name, learning_rate=0.001):
    new_model = clone_model(base_model)
    new_model.set_weights(base_model.get_weights())

    freeze = True 
    reinitialize = False

    for layer in new_model.layers:
        if layer.name == freeze_up_to_layer_name:
            freeze = False  # Stop freezing layers from this point onwards
            reinitialize = True  # Start reinitializing the subsequent layers
        layer.trainable = not freeze

        # If reinitialization is flagged, reinitialize the layer's weights
        if reinitialize and hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            # Get initializers
            kernel_initializer = layer.kernel_initializer
            bias_initializer = layer.bias_initializer
            
            # Reinitialize weights
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(kernel_initializer(shape=layer.kernel.shape))
            if hasattr(layer, 'bias'):
                layer.bias.assign(bias_initializer(shape=layer.bias.shape))

    new_model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mae")

    return new_model

class StepResult(models.Model):
    steps_in_future = models.IntegerField(default=0)
    cluster = models.ForeignKey(StockCluster, on_delete=models.CASCADE, related_name='cluster_results')
    dir_accuracy = models.FloatField(default=0)
    weighted_dir_acc = models.FloatField(default=0)
    predicted_return = models.FloatField(default=0)
    actual_return = models.FloatField(default=0)
    train_set_length = models.IntegerField(default=0)
    test_set_length = models.IntegerField(default=0)
    p_l = models.FloatField(default=0)
    predicted_values = models.JSONField(default=list)
    actual_values = models.JSONField(default=list)

    def get_results(self):
        step_results = {
            "steps_in_future": self.steps_in_future,
            "dir_accuracy": self.dir_accuracy,
            "weighted_dir_acc": self.weighted_dir_acc,
            "predicted_return": self.predicted_return,
            "actual_return": self.actual_return,
        }

        return step_results

def create_attention(input_shape, output_steps, num_features):
    # Encoder
    encoder_inputs = Input(shape=(None, input_shape))
    encoder_lstm1 = LSTM(units=25, return_sequences=True, activation='tanh')(encoder_inputs)
    encoder_lstm2, state_h, state_c = LSTM(units=50, return_sequences=True, return_state=True, activation='tanh')(encoder_lstm1)

    # Attention mechanism
    attention = tf.keras.layers.Attention()
    # Decoder
    # Initialize LSTM decoder layer
    decoder_lstm = LSTM(units=50, return_sequences=False, return_state=True, activation='tanh')


    all_decoder_outputs = []
    decoder_input = tf.zeros_like(encoder_inputs[:, 0, :])  # Initial decoder input, zeros
    decoder_dense = TimeDistributed(Dense(num_features))

    # Initial states for the decoder LSTM
    decoder_states = [state_h, state_c]

    for _ in range(output_steps):
        # Prepare a query for the attention layer
        # Use a dense layer to transform the decoder state to the correct shape
        query_dense = Dense(50)  # Transform to match the encoder output dimension
        query = query_dense(decoder_states[0])  # Apply transformation to hidden state
        query = tf.expand_dims(query, 1)  # Expand dims to fit attention layer

        # Computing a context vector using attention mechanism
        context_vector, attention_weights = attention([query, encoder_lstm2], return_attention_scores=True)

        # Pass the context vector and previous states to LSTM decoder
        decoder_output, state_h, state_c = decoder_lstm(context_vector, initial_state=decoder_states)
        
        # Reshape decoder output and store
        decoder_output = tf.expand_dims(decoder_output, 1)
        decoder_output = decoder_dense(decoder_output)
        all_decoder_outputs.append(decoder_output)

        # Update the decoder input for the next timestep
        decoder_input = tf.squeeze(decoder_output, axis=1)
        decoder_states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Concatenate(axis=1)(all_decoder_outputs)

    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

        
