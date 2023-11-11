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

class ClusterGroupParams: 
    def __init__(self, start_date,end_date = None): 
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")

        self.end_date = end_date
        if self.end_date is None:
            self.end_date = datetime.today().date()

        self.X_feature_dict = None 
        self.y_feature_dict = None
        self.X_cols = None
        self.y_cols = None
        self.X_feature_sets = None
        self.y_feature_sets = None
        self.train_seq_elements = None
        self.test_seq_elements = None

class StockClusterGroupParams(ClusterGroupParams): 

    def __init__(self, start_date, tickers, target_cols, n_steps, cluster_features, interval = '1d', end_date = None): 
        super().__init__(start_date, end_date)
        self.tickers = tickers
        self.target_cols = target_cols
        self.n_steps = n_steps
        self.interval = interval
        self.cluster_features = cluster_features

        self.scaling_dict = {
            'price_vars': ScalingMethod.SBSG,
            'trend_vars' : ScalingMethod.SBS,
            'pctChg_vars' : ScalingMethod.QUANT_MINMAX,
            'rolling_vars' : ScalingMethod.QUANT_MINMAX_G,
            'target_vars' : ScalingMethod.UNSCALED
        }



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
    



class ClusterGroup: 
    def __init__(self,group_params = None): 
        self.group_params = group_params
    
    def set_group_params(self,group_params): 
        self.group_params = group_params

    def create_data_set(self): 
        pass

    def create_sequence_set(self):
        pass

class StockClusterGroup(ClusterGroup):
    def __init__(self,group_params = None): 
        super().__init__(group_params=group_params)
        self.data_set = None
        self.sequence_set = None

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
            self.clusters.append(StockCluster(self,cur_train_seq_elements,cur_test_seq_elements))


    def display_all_clusters(self):
        for cluster in self.clusters:
            fig = cluster.visualize_cluster()
            fig.show()
    
    def train_all_rnns(self,model_features, fine_tune = True):
        model = None 
        if fine_tune:
            self.train_general_model(model_features)
            model = self.general_model
         
        for cluster in self.clusters:
            cluster.train_rnn(model_features,model)
    
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


class Cluster:
    def __init__(self, cluster_group,train_seq_elements,test_seq_elements):
        self.cluster_group = cluster_group
        self.train_seq_elements = train_seq_elements
        self.test_seq_elements = test_seq_elements
    
    def get_3d_array(self):
        self.X_train, self.y_train = SequenceElement.create_array(self.train_seq_elements)
        self.X_test, self.y_test = SequenceElement.create_array(self.test_seq_elements)
        return self.X_train, self.y_train, self.X_test, self.y_test

class StockCluster(Cluster):
    def __init__(self,cluster_group, train_seq_elements, test_seq_elements):
        super().__init__(cluster_group,train_seq_elements,test_seq_elements)
        self.cluster_group = cluster_group
        self.train_seq_elements = train_seq_elements
        self.test_seq_elements = test_seq_elements
        self.label = train_seq_elements[0].cluster_label
        self.get_3d_array()
    
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
        for i in range(num_days):
            same_day = ((results[f'predicted_{i+1}'] > 0) & (results[f'real_{i+1}'] > 0)) | \
                    ((results[f'predicted_{i+1}'] < 0) & (results[f'real_{i+1}'] < 0))
            accuracy = round(same_day.mean() * 100,2)
            w_accuracy = round(weighted_dir_acc(results[f'predicted_{i+1}'], results[f'real_{i+1}']),2)

            output_string += (
                f"Accuracy{i+1}D {accuracy}% (Weighted: {w_accuracy}%) "
                f"PredictedRet: {results[f'predicted_{i+1}'].mean()} "
                f"ActRet: {results[f'real_{i+1}'].mean()}\n"
            )
        
        output_string += f"Train set length: {len(X_train_filtered)} Test set length: {len(y_test)}\n"

        with open('output.txt', 'a') as f:
            f.write(output_string)







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

    



        
