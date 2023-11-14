from datetime import datetime
from .SequencePreprocessing import StockSequenceSet, SequenceElement, ScalingMethod
from .TSeriesPreproccesing import StockDataSet
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import plotly.graph_objects as go
import math
from keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU,BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pickle
from django.db import models
import os


class SupportedParams(models.Model):
    """
    Class to contain potential parameters for running the pipeline
    """
    features = models.JSONField(default=list)
    name = models.CharField(max_length=100,default="")

    def generate_features(self):
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

class ClusterGroupParams(models.Model): 
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
        self.X_feature_sets = None
        self.y_feature_sets = None
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

        # self.scaling_dict = {
        #     'price_vars': ScalingMethod.SBSG,
        #     'trend_vars' : ScalingMethod.SBS,
        #     'pctChg_vars' : ScalingMethod.QUANT_MINMAX,
        #     'rolling_vars' : ScalingMethod.QUANT_MINMAX_G,
        #     'target_vars' : ScalingMethod.UNSCALED
        # }
    


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
    n_clusters = models.IntegerField(default=0)
    class Meta:
        abstract = True
        

    def set_group_params(self,group_params): 
        self.group_params = group_params

    def create_data_set(self): 
        pass

    def create_sequence_set(self):
        pass

class StockClusterGroup(ClusterGroup):
    group_params = models.OneToOneField(StockClusterGroupParams, on_delete=models.CASCADE, related_name='group_params')

    def create_data_set(self):
        self.data_set = StockDataSet(self.group_params)
        self.data_set.preprocess_pipeline() 
        self.group_params = self.data_set.group_params

    
    def create_sequence_set(self):
        self.sequence_set = self.data_set.create_sequence_set() 
        self.sequence_set.preprocess_pipeline(add_cuma_pctChg_features=True)
        self.group_params = self.sequence_set.group_params
    
    def run_clustering(self,alg = 'TSKM',metric = "euclidean"):
        
        self.get_3d_array()
        X_train_cluster = self.filter_by_features(self.X_train, self.group_params.cluster_features)
        X_test_cluster = self.filter_by_features(self.X_test, self.group_params.cluster_features)

        print(X_train_cluster.shape)
        print(X_test_cluster.shape)



        if alg == 'TSKM':
            # n_clusters = self.determine_n_clusters(X_train_cluster,metric)
            n_clusters = math.ceil(math.sqrt(len(X_train_cluster))) // 3
            self.cluster_alg = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,random_state=3)
        
        self.train_labels = self.cluster_alg.fit_predict(X_train_cluster)
        self.test_labels = self.cluster_alg.predict(X_test_cluster)

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        if len(self.train_labels) != len(train_seq_elements):
            raise ValueError("The number of labels does not match the number of sequences")
        if len(self.test_labels) != len(test_seq_elements):
            raise ValueError("The number of labels does not match the number of sequences")

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

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        train_labels = np.unique([x.cluster_label for x in train_seq_elements])

        self.clusters = [] 

        for label in train_labels:
            cur_train_seq_elements = [x for x in train_seq_elements if x.cluster_label == label]
            cur_test_seq_elements = [x for x in test_seq_elements if x.cluster_label == label]

            if len(cur_train_seq_elements) == 0 or len(cur_test_seq_elements) == 0:
                continue

            cluster = StockCluster.objects.create(label=label, cluster_group=self)
            cluster.initialize(cur_train_seq_elements,cur_test_seq_elements)
            self.clusters.append(cluster)


    # def display_all_clusters(self):
    #     for cluster in self.clusters:
    #         fig = cluster.visualize_cluster()
    #         fig.show()
    
    def train_all_rnns(self,model_features, fine_tune = True):
        model = None 
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
            else: 
                cluster.delete()

        self.group_params.save() 

    
    def train_general_model(self,model_features):
        X_train = self.filter_by_features(self.X_train, model_features)
        X_test = self.filter_by_features(self.X_test, model_features)
        y_train = self.y_train
        y_test = self.y_test

        self.general_model = create_model(len(model_features))
        
        self.general_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)



    def filter_by_features(self,seq, feature_list):
        seq = seq.copy()
        indices = [self.group_params.X_feature_dict[x] for x in feature_list]
        # Using numpy's advanced indexing to select the required features
        return seq[:, :, indices]
    
    def get_3d_array(self): 
        self.X_train, self.y_train, self.X_test, self.y_test = self.sequence_set.get_3d_array()
        return self.X_train, self.y_train, self.X_test, self.y_test


class Cluster(models.Model):
    label = models.IntegerField(default=-1)
    elements_file_string = models.CharField(max_length=100,default="")
    model_file_string = models.CharField(max_length=100,default="")
    class Meta:
        abstract = True
    
    def initialize(self,train_seq_elements,test_seq_elements):
        self.train_seq_elements = train_seq_elements
        self.test_seq_elements = test_seq_elements
        self.get_3d_array()
        
    def get_3d_array(self):
        if len(self.train_seq_elements) == 0 or len(self.test_seq_elements) == 0:
            raise ValueError("No sequences in this cluster")
        
        self.X_train, self.y_train = SequenceElement.create_array(self.train_seq_elements)
        self.X_test, self.y_test = SequenceElement.create_array(self.test_seq_elements)
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def serialize(self):
        group_name = self.cluster_group.group_params.name
        item_dict = {
            "train_seq_elements": self.train_seq_elements,
            "test_seq_elements": self.test_seq_elements,
        }
        self.elements_file_string = f"SavedModels/{group_name}/Cluster{self.label}Elements.pkl"
        self.model_file_string = f"SavedModels/{group_name}/Cluster{self.label}Model.pkl"


        directory1 = os.path.dirname(self.elements_file_string)
        directory2 = os.path.dirname(self.model_file_string)

        # Check if the directory exists
        if not os.path.exists(directory1):
            # Create the directory if it doesn't exist
            os.makedirs(directory1)

        with open(directory1, 'wb') as f:
            pickle.dump(item_dict, f)
        with open(directory2, 'wb') as f:
            pickle.dump(self.model, f)
    
    def deserialize_elements(self):
        with open(self.elements_file_string, 'rb') as f:
            item_dict = pickle.load(f)
        
        self.train_seq_elements = item_dict["train_seq_elements"]
        self.test_seq_elements = item_dict["test_seq_elements"]

        self.get_3d_array()
    
    def deserialize_model(self):
        with open(self.model_file_string, 'rb') as f:
            self.model = pickle.load(f)

class StockCluster(Cluster):
    cluster_group = models.ForeignKey(StockClusterGroup, on_delete=models.CASCADE, related_name='clusters_obj')
    
    def remove_outliers(self):
        pass

    def visualize_cluster(self, isTrain = True, y_range = [-5,5]):
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
    
    def visualize_target_values(self):
        '''
        Create a scatter plot of the target values for the cluster using Plotly
        '''
        target_vals = self.y_train

        num_elements = len(target_vals)
        num_steps = len(target_vals[0])

        # Create traces for each step
        traces = []
        for step in range(num_steps):
            # Create a scatter trace for this step
            scatter = go.Scatter(
                x=[step+1] * num_elements,
                y=target_vals[:, step],
                mode='markers',
                name=f'Step {step+1}'
            )
            traces.append(scatter)

        # Calculate averages and create a trace for the averages
        averages = np.mean(target_vals, axis=0)
        averages_trace = go.Scatter(
            x=list(range(1, num_steps + 1)),
            y=averages,
            mode='lines+markers',
            name='Average',
            line=dict(color='red', dash='dash', width=2),
            marker=dict(size=12)
        )
        traces.append(averages_trace)

        # Create a layout
        layout = go.Layout(
            title='Target Values Scatter Plot',
            xaxis=dict(title='Step'),
            yaxis=dict(title='Value', range=[-10, 10]),
            showlegend=True
        )

        # Create a figure
        fig = go.Figure(data=traces, layout=layout)

        return fig


    
    def train_rnn(self,model_features,general_model = None):
        if len(self.X_train) == 0 or len(self.X_test) == 0:
            return
        X_train_filtered = self.cluster_group.filter_by_features(self.X_train, model_features)
        X_test_filtered = self.cluster_group.filter_by_features(self.X_test, model_features)
        y_train = self.y_train
        y_test = self.y_test

        if general_model is None: 
            self.model = create_model(len(model_features))
        else: 
            freeze_up_to_layer_name = 'repeat_vector'
            self.model = clone_for_tuning(general_model, freeze_up_to_layer_name, learning_rate=0.0001)
        
        self.model.fit(X_train_filtered, y_train, epochs=75, batch_size=16, validation_data=(X_test_filtered, y_test), verbose=1)

        predicted_y = self.model.predict(X_test_filtered)
        predicted_y = np.squeeze(predicted_y, axis=-1)

        num_days = predicted_y.shape[1]  # Assuming this is the number of days
        results = pd.DataFrame(predicted_y, columns=[f'predicted_{i+1}' for i in range(num_days)])

        
        for i in range(num_days):
            results[f'real_{i+1}'] = y_test[:, i]

        # Generate output string with accuracies

        output_string = f"Cluster Number: {self.label}\n"
        self.step_results = [] 
        for i in range(num_days):
            step_result = StepResult.objects.create(steps_in_future = i+1, cluster = self,train_set_length = len(X_train_filtered), test_set_length = len(y_test))
            same_day = ((results[f'predicted_{i+1}'] > 0) & (results[f'real_{i+1}'] > 0)) | \
                    ((results[f'predicted_{i+1}'] < 0) & (results[f'real_{i+1}'] < 0))
            accuracy = round(same_day.mean() * 100,2)
            w_accuracy = round(weighted_dir_acc(results[f'predicted_{i+1}'], results[f'real_{i+1}']),2)

            step_result.dir_accuracy = accuracy
            step_result.weighted_dir_acc = w_accuracy
            step_result.save()

            output_string += (
                f"Accuracy{i+1}D {accuracy}% (Weighted: {w_accuracy}%) "
                f"PredictedRet: {results[f'predicted_{i+1}'].mean()} "
                f"ActRet: {results[f'real_{i+1}'].mean()}\n"
            )
        
        output_string += f"Train set length: {len(X_train_filtered)} Test set length: {len(y_test)}\n"

        with open('output.txt', 'a') as f:
            f.write(output_string)

    
    def filter_results(self,threshhold = 0.2,test_set_length = 30):
        self.step_results = StepResult.objects.filter(cluster = self)
        for result in self.step_results:
            if result.dir_accuracy < threshhold or result.test_set_length < test_set_length:
                result.delete()
        self.num_results = len(self.step_results)
    
    def generate_results_string(self):
        string = "Train Set Length: " + str(len(self.X_train)) + " Test Set Length: " + str(len(self.y_test)) + "\n"
        self.step_results = self.cluster_results.all()
        for result in self.step_results:
            string += result.get_results_string()
        return string
    






def weighted_dir_acc(predicted, actual):
    directional_accuracy = (np.sign(predicted) == np.sign(actual)).astype(int)
    magnitude_difference = np.abs(np.abs(predicted) - np.abs(actual)) + 1e-6
    weights = np.abs(actual) / magnitude_difference
    return np.sum(directional_accuracy * weights) / np.sum(weights) * 100

def create_model(input_shape):
    model_lstm = Sequential()
    
    # Encoder
    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True, input_shape=(None, input_shape), name='encoder_lstm_1'))
    model_lstm.add(BatchNormalization(name='encoder_bn_1'))
    model_lstm.add(Dropout(0.2, name='encoder_dropout_1'))

    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True, name='encoder_lstm_2'))
    model_lstm.add(BatchNormalization(name='encoder_bn_2'))
    model_lstm.add(Dropout(0.2, name='encoder_dropout_2'))

    model_lstm.add(LSTM(units=100, activation='tanh', return_sequences=True, name='encoder_lstm_3'))
    model_lstm.add(BatchNormalization(name='encoder_bn_3'))
    model_lstm.add(Dropout(0.2, name='encoder_dropout_3'))

    model_lstm.add(LSTM(units=100, activation='tanh', name='encoder_lstm_4'))
    model_lstm.add(BatchNormalization(name='encoder_bn_4'))
    model_lstm.add(Dropout(0.2, name='encoder_dropout_4'))
    
    # Repeat Vector
    model_lstm.add(RepeatVector(6, name='repeat_vector'))  # Assuming you are predicting for 6 steps

    # Decoder
    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True, name='decoder_lstm_1'))
    model_lstm.add(BatchNormalization(name='decoder_bn_1'))
    model_lstm.add(Dropout(0.2, name='decoder_dropout_1'))

    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True, name='decoder_lstm_2'))
    model_lstm.add(BatchNormalization(name='decoder_bn_2'))
    model_lstm.add(Dropout(0.2, name='decoder_dropout_2'))
    
    model_lstm.add(TimeDistributed(Dense(1), name='time_distributed_output'))

    # Compile
    optimizer = Adam(learning_rate=0.001)
    model_lstm.compile(optimizer=optimizer, loss="mae")

    return model_lstm


def clone_for_tuning(base_model, freeze_up_to_layer_name, learning_rate=0.0001):
    new_model = clone_model(base_model)
    new_model.set_weights(base_model.get_weights())

    freeze = True 

    for layer in new_model.layers:
        if layer.name == freeze_up_to_layer_name:
            freeze = False
        layer.trainable = not freeze

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

    def get_results_string(self):
        return str(self.steps_in_future) +"Step, " + str(self.dir_accuracy) + "% Accuracy, " + str(self.weighted_dir_acc) + "% Weighted Accuracy, " + str(self.predicted_return) + " Predicted Return, " + str(self.actual_return) + " Actual Return\n"




        
