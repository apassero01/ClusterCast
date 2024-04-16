from datetime import datetime
from ClusterPipeline.models.SequencePreprocessing import (
    StockSequenceSet,
    SequenceElement,
    ScalingMethod,
)
from ClusterPipeline.models.TSeriesPreproccesing import StockDataSet
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import plotly.graph_objects as go
import math
from keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    GRU,
    BatchNormalization,
    Input,
    Concatenate,
    Attention,
    Masking,
)
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
from ClusterPipeline.models.RNNModels import RNNModel, ModelTypes, StepResult
from tensorflow.keras.initializers import glorot_uniform, zeros
from django.dispatch import receiver
import shutil
import gc
import random

class SupportedParams(models.Model):
    """
    Class to contain potential parameters for running the pipeline
    We can run methods in this class to add parameters to the database that can be used for front end forms
    """

    features = models.JSONField(default=list)
    name = models.CharField(max_length=100, default="")

    pct_chg_features = models.JSONField(default=list)
    rolling_features = models.JSONField(default=list)
    trend_features = models.JSONField(default=list)
    price_features = models.JSONField(default=list)
    cuma_features = models.JSONField(default=list)
    lag_features = models.JSONField(default=list)
    momentum_features = models.JSONField(default=list)

    def generate_features(self):
        """
        Method to generate a list of all the features that the current pipeline produces
        """
        tickers = ["spy"]
        start = "2020-01-01"
        target_cols = [
            "sumpctChgclose_1",
            "sumpctChgclose_2",
            "sumpctChgclose_3",
            "sumpctChgclose_4",
            "sumpctChgclose_5",
            "sumpctChgclose_6",
        ]
        n_steps = 2
        interval = "1d"
        cluster_features = ["pctChgclose_cumulative"]
        group_params = StockClusterGroupParams(
            start_date=start,
            tickers=tickers,
            interval=interval,
            target_cols=target_cols,
            n_steps=n_steps,
            cluster_features=cluster_features,
        )
        group_params.initialize()
        group_params.scaling_dict = {
            "price_vars": ScalingMethod.UNSCALED,
            "trend_vars": ScalingMethod.UNSCALED,
            "pctChg_vars": ScalingMethod.STANDARD,
            "rolling_vars": ScalingMethod.STANDARD,
            "target_vars": ScalingMethod.QUANT_MINMAX,
            "lag_feature_vars": ScalingMethod.STANDARD,
            "momentum_vars": ScalingMethod.STANDARD,
        }
        cluster_group = StockClusterGroup()
        cluster_group.set_group_params(group_params)
        cluster_group.create_data_set(to_train=False)
        cluster_group.create_sequence_set()

        self.features = list(cluster_group.sequence_set.group_params.X_cols)

    def generate_features_by_type(self):
        """
        Similiar types of features are grouped together. This method generates a list of features for each type
        """
        tickers = ["spy"]
        start = "2020-01-01"
        target_cols = [
            "sumpctChgclose_1",
            "sumpctChgclose_2",
            "sumpctChgclose_3",
            "sumpctChgclose_4",
            "sumpctChgclose_5",
            "sumpctChgclose_6",
        ]
        n_steps = 2
        interval = "1d"
        cluster_features = ["pctChgclose_cumulative"]
        group_params = StockClusterGroupParams(
            start_date=start,
            tickers=tickers,
            interval=interval,
            target_cols=target_cols,
            n_steps=n_steps,
            cluster_features=cluster_features,
        )
        group_params.initialize()
        group_params.scaling_dict = {
            "price_vars": ScalingMethod.UNSCALED,
            "trend_vars": ScalingMethod.UNSCALED,
            "pctChg_vars": ScalingMethod.STANDARD,
            "rolling_vars": ScalingMethod.STANDARD,
            "target_vars": ScalingMethod.QUANT_MINMAX,
            "lag_feature_vars": ScalingMethod.STANDARD,
            "momentum_vars": ScalingMethod.STANDARD,
        }
        cluster_group = StockClusterGroup()
        cluster_group.set_group_params(group_params)
        cluster_group.create_data_set(to_train=False)
        cluster_group.create_sequence_set()

        self.pct_chg_features = next(
            filter(
                lambda feature_set: feature_set.name == "pctChg_vars",
                cluster_group.group_params.X_feature_sets,
            )
        ).cols

        self.lag_features = next(
            filter(
                lambda feature_set: "lag_features_vars" in feature_set.name,
                cluster_group.group_params.X_feature_sets,
            )
        ).cols

        self.momentum_features = next(
            filter(
                lambda feature_set: "momentum_vars" in feature_set.name,
                cluster_group.group_params.X_feature_sets,
            )
        ).cols

        # self.cuma_features = next(
        #     (
        #         filter(
        #             lambda feature_set: "cum" in feature_set.name,
        #             cluster_group.group_params.X_feature_sets,
        #         )
        #     )
        # ).cols
        self.price_features = next(
            (
                filter(
                    lambda feature_set: "price" in feature_set.name,
                    cluster_group.group_params.X_feature_sets,
                )
            )
        ).cols
        self.trend_features = next(
            (
                filter(
                    lambda feature_set: "trend" in feature_set.name,
                    cluster_group.group_params.X_feature_sets,
                )
            )
        ).cols

        rolling_feat = list(
            (
                filter(
                    lambda feature_set: "rolling" in feature_set.name,
                    cluster_group.group_params.X_feature_sets,
                )
            )
        )

        self.rolling_features = []
        for feature in rolling_feat:
            self.rolling_features += feature.cols

    
    def remove_features(self, feature_sub_category): 
        '''
        Method to remove a feature from the list of features
        '''
        tickers = ['spy']
        start = '2020-01-01'
        target_cols = ['sumpctChgclose_1','sumpctChgclose_2','sumpctChgclose_3','sumpctChgclose_4','sumpctChgclose_5','sumpctChgclose_6']
        n_steps = 2
        interval = '1d'
        cluster_features = ['pctChgclose_cumulative']
        group_params = StockClusterGroupParams(start_date = start, tickers = tickers, interval = interval, target_cols = target_cols, n_steps = n_steps,cluster_features = cluster_features)
        group_params.initialize()
        group_params.scaling_dict = {
                    'price_vars': ScalingMethod.SBSG,
                    'trend_vars' : ScalingMethod.SBS,
                    'pctChg_vars' : ScalingMethod.QUANT_MINMAX,
                    'rolling_vars' : ScalingMethod.QUANT_MINMAX_G,
                    'target_vars' : ScalingMethod.UNSCALED
                    }
        cluster_group = StockClusterGroup()
        cluster_group.set_group_params(group_params)
        cluster_group.create_data_set(to_train=False)
        cluster_group.create_sequence_set()

        X_feature_sets = cluster_group.group_params.X_feature_sets
        for feature_set in X_feature_sets:
            for key in feature_set.sub_categories.keys():
                if key == feature_sub_category:
                    features_to_remove = feature_set.sub_categories[key]
        
        all_models = RNNModel.objects.all()

        models_to_delete = []

        for model in all_models:
            model_features = model.model_features
            for feature in features_to_remove:
                if feature in model_features:
                    models_to_delete.append(model)
                    break
        
        for model in models_to_delete:
            model.delete()

class ClusterGroupParams(models.Model): 
    '''
    Class to contain parameters for running the pipeline. This class is abstract
    '''

    start_date = models.DateField()
    end_date = models.DateField()
    n_steps = models.IntegerField()
    scaling_dict = models.JSONField(default=dict)
    name = models.CharField(max_length=400, default="")

    class Meta:
        abstract = True

    def initialize(self):
        self.X_cols = None
        self.y_cols = None
        self.data_sets = []
        self.train_seq_elements = None
        self.test_seq_elements = None
        self.X_feature_sets = []
        self.y_feature_sets = []

    def set_scaling_dict(self, scaling_dict):
        self.scaling_dict = {key: value.value for key, value in scaling_dict.items()}

    def get_scaling_dict(self):
        return {key: ScalingMethod(value) for key, value in self.scaling_dict.items()}


class StockClusterGroupParams(ClusterGroupParams):
    tickers = models.JSONField(default=list)
    target_cols = models.JSONField(default=list)
    interval = models.CharField(max_length=10)
    cluster_features = models.JSONField(default=list, blank=True)
    training_features = models.JSONField(default=list, blank=True)
    model_params = models.JSONField(default=list, blank=True)

    def initialize(self):
        super().initialize()
        self.name = (
            str(self.tickers)
            + str(self.start_date)
            + "-"
            + str(self.end_date)
            + "-"
            + str(self.n_steps)
            + "steps"
            + "-"
            + str(self.interval)
            + "-"
            + str(self.cluster_features)
            + "-"
            + str(self.pk)
        )

    def create_model_dir(self):
        model_dir = f"SavedModels/{self.name}/"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            counter = 1
            while os.path.exists(model_dir):
                self.name = self.name + str(counter)
                model_dir = f"SavedModels/{self.name}/"
                counter += 1


class ClusterPeriod:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.current_cluster_group = None

    def create_cluster_group(self, group_params):
        pass


class StockClusterPeriod(ClusterPeriod):
    def __init__(self, start_date, end_date):
        super().__init__(start_date, end_date)

    def create_cluster_params(
        self, tickers, target_cols, n_steps, cluster_features, interval="1d"
    ):
        return StockClusterGroupParams(
            self.start_date,
            tickers,
            target_cols,
            n_steps,
            cluster_features,
            interval,
            self.end_date,
        )

    def update_param_scaling_dict(self, group_params, scaling_dict):
        group_params.scaling_dict = scaling_dict

    def create_cluster_group(self, group_params):
        cluster_group = StockClusterGroup(group_params)
        cluster_group.create_data_set()
        cluster_group.create_sequence_set()

        self.current_cluster_group = cluster_group

    def cluster_current_group(self, alg="TSKM", metric="euclidean"):
        if self.cluster_current_group is None:
            raise ValueError("No cluster group has been created yet")

        self.current_cluster_group.run_clustering(alg, metric)
        self.current_cluster_group.create_clusters()

    def display_current_group(self):
        if self.current_cluster_group is None:
            raise ValueError("No cluster group has been created yet")
        self.current_cluster_group.display_all_clusters()


class ClusterGroup(models.Model):
    """
    Class to encapsulate a group of clusters ie. The class that can run a clustering algorithm on all the data points, and
    then create cluster objects for each label returned from the clustering algorithm. This class is abstract.
    """

    n_clusters = models.IntegerField(default=0)
    train_labels = models.JSONField(default=list)
    test_labels = models.JSONField(default=list)

    class Meta:
        abstract = True

    def set_group_params(self, group_params):
        self.group_params = group_params

    def create_data_set(self):
        pass

    def create_sequence_set(self):
        pass


class StockClusterGroup(ClusterGroup):
    """
    Implementation of the ClusterGroup class for stock data
    """

    group_params = models.OneToOneField(
        StockClusterGroupParams, on_delete=models.CASCADE, related_name="group_params"
    )

    def create_data_set(self, to_train=True):
        """
        Method to create a StockDataSet object from the group_params
        """
        data_sets = []
        for ticker in self.group_params.tickers:
            data_set = StockDataSet(self.group_params, ticker)
            data_set.preprocess_pipeline(to_train=to_train)
            data_sets.append(data_set)
        self.group_params = data_sets[0].group_params

        self.group_params.data_sets = data_sets

    def create_sequence_set(self):
        """
        Method to create a StockSequenceSet object from the data_set
        """

        self.sequence_set = StockSequenceSet(self.group_params)
        self.sequence_set.preprocess_pipeline(add_cuma_pctChg_features=False)
        self.group_params = self.sequence_set.group_params
        self.X_feature_dict = self.group_params.X_feature_dict
        self.y_feature_dict = self.group_params.y_feature_dict

    def run_clustering(self, alg="TSKM", metric="euclidean"):
        """
        Method to run a clustering algorithm on the data set.

        Parameters:
        alg: The clustering algorithm to use. Currently only supports TimeSeriesKMeans
        metric: The metric to use for the clustering algorithm. default is euclidean
        """
        self.get_3d_array()
        X_train_cluster = self.filter_by_features(
            self.X_train, self.group_params.cluster_features, self.X_feature_dict
        )
        X_test_cluster = self.filter_by_features(
            self.X_test, self.group_params.cluster_features, self.X_feature_dict
        )

        if alg == "TSKM":
            # n_clusters = self.determine_n_clusters(X_train_cluster,metric)
            n_clusters = math.ceil(math.sqrt(len(X_train_cluster))) // 7
            # n_clusters = 1
            self.cluster_alg = TimeSeriesKMeans(
                n_clusters=n_clusters, metric=metric, random_state=3
            )

        self.train_labels = list(self.cluster_alg.fit_predict(X_train_cluster))
        self.test_labels = list(self.cluster_alg.predict(X_test_cluster))

        self.train_labels = [int(x) for x in self.train_labels]
        self.test_labels = [int(x) for x in self.test_labels]

        self.cluster_distances = self.cluster_alg.transform(X_train_cluster)
        self.cluster_centers = self.cluster_alg.cluster_centers_

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        if len(self.train_labels) != len(train_seq_elements):
            raise ValueError(
                "The number of labels does not match the number of sequences"
            )
        if len(self.test_labels) != len(test_seq_elements):
            raise ValueError(
                "The number of labels does not match the number of sequences"
            )

        # np.random.shuffle(self.train_labels)
        # np.random.shuffle(self.test_labels)

        for i in range(len(train_seq_elements)):
            seq_element = train_seq_elements[i]
            seq_element.cluster_label = self.train_labels[i]
        for i in range(len(test_seq_elements)):
            seq_element = test_seq_elements[i]
            seq_element.cluster_label = self.test_labels[i]

    def determine_n_clusters(self, X_train_cluster, metric="euclidean"):
        """
        Method that utilizes the elbow method to determine the optimal number of clusters
        """
        min_clusters = math.ceil(math.sqrt(len(X_train_cluster))) // 8
        max_clusters = math.ceil(math.sqrt(len(X_train_cluster))) // 2

        wcss = []
        self.K = range(min_clusters, max_clusters, 4)
        for k in self.K:
            cluster_alg = TimeSeriesKMeans(n_clusters=k, metric=metric, random_state=3)
            train_labels = cluster_alg.fit_predict(X_train_cluster)
            wcss.append(cluster_alg.inertia_)

        kn = KneeLocator(self.K, wcss, curve="convex", direction="decreasing")

        self.wcss = wcss
        return kn.knee

    def create_clusters(self):
        """
        Method to create a StockCluster object for each label returned from the clustering algorithm
        """

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        train_labels_unique = np.unique([self.train_labels])

        self.clusters = []

        for label in train_labels_unique:
            # Get all the sequences that belong to this cluster
            cur_train_seq_elements = [
                x for x in train_seq_elements if x.cluster_label == label
            ]
            cur_test_seq_elements = [
                x for x in test_seq_elements if x.cluster_label == label
            ]

            # Get the distances of all the sequences in this cluster to the cluster center
            cluster_distances = self.cluster_distances[
                self.train_labels == label, label
            ]

            std_dev = np.std(cluster_distances)

            iqr = np.percentile(cluster_distances, 75) - np.percentile(
                cluster_distances, 25
            )

            centroid = self.cluster_centers[label].tolist()

            metrics = {"std_dev": std_dev, "iqr": iqr}

            if len(cur_train_seq_elements) == 0 or len(cur_test_seq_elements) == 0:
                continue

            # Create the object and pass in the label, cluster_group and associated metrics
            cluster = StockCluster.objects.create(
                label=label,
                cluster_group=self,
                cluster_metrics=metrics,
                centroid=centroid,
            )
            cluster.initialize(
                cur_train_seq_elements,
                cur_test_seq_elements,
                self.group_params.X_feature_dict,
                self.group_params.y_feature_dict,
            )
            self.clusters.append(cluster)

    def manually_create_cluster(self, train_seq_elements, test_seq_elements):
        """
        Method to manually create a cluster. This method is used when the user wants to create a cluster manually
        """
        cluster = StockCluster.objects.create(
            label = 1,
            cluster_group = self,
            cluster_metrics = {},
            centroid = [],
        )
        cluster.initialize(
            train_seq_elements,
            test_seq_elements,
            self.group_params.X_feature_dict,
            self.group_params.y_feature_dict,
        )

        self.clusters.append(cluster)


    def train_all_rnns(self, model_features, training_dict, fine_tune=False, model = None):
        """
        Method to train an RNN for each cluster. The functionality for training an RNN is encapsulated in the train_rnn method of the StockCluster class.
        This method iterates over the clusters and trains the model.

        Parameters:
        model_features: The features to use for training the RNN.
        fine_tune: Boolean to indicate whether we are training a base model and fine tunining it on the individual clusters.
        """

        for cluster in self.clusters:
            cluster.train_rnn(model_features, self.group_params.target_cols, training_dict = training_dict, num_feauture_iterations = self.group_params.feature_sample_num, sample_size=self.group_params.feature_sample_size )
            cluster.save()
            # Maybe delete the models later
            cluster.hard_filter_models()
            del cluster
            gc.collect()

        self.group_params.save()

    def train_single_cluster(self, cluster, training_dict):
        """
        Method to train an RNN for a single cluster. This method is used for fine tuning on the individual clusters.
        """
        model_features = self.group_params.training_features
        cluster.group_params = self.group_params
        cluster.train_rnn(model_features, self.group_params.target_cols, training_dict = training_dict, num_feauture_iterations = 2, sample_size=20 )
        cluster.save()
        cluster.hard_filter_models(accuracy_threshold=30, epoch_threshold=1)

        self.group_params.save()

    def filter_by_features(self, seq, feature_list, X_feature_dict):
        """
        Method to filter a 3d array of sequences by a list of features.
        """
        indices = [X_feature_dict[x] for x in feature_list]
        # Using numpy's advanced indexing to select the required features
        return seq[:, :, indices]
    
    def filter_y_by_features(self,seq, feature_list, y_feature_dict):
        '''
        Method to filter a 3d array of sequences by a list of features.
        '''
        indices = [y_feature_dict[x] for x in feature_list]
        # Using numpy's advanced indexing to select the required features
        return seq[:, indices]

    def get_3d_array(self):
        """
        Method to get the 3d array of sequences from the sequence set
        """
        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.sequence_set.get_3d_array()
        return self.X_train, self.y_train, self.X_test, self.y_test

    def load_saved_clusters(self):
        """
        Method to load saved clusters from the database
        """
        self.clusters = StockCluster.objects.filter(cluster_group=self)

    def generate_new_group(self, training_dict):
        """
        Method to generate a new group. This method is used when the user wants to run the pipeline from scratch
        """
        model_features = self.group_params.training_features
        self.create_data_set()
        self.create_sequence_set()
        self.run_clustering()
        self.create_clusters()
        self.train_all_rnns(model_features=model_features, training_dict=training_dict)

        self.save()

    def load_saved_group(self):
        """
        Method to load a saved group from the database. We perform the preprocessing steps that are necessary and avoid the ones that are not
        """
        self.group_params.initialize()
        self.create_data_set()
        self.create_sequence_set()
        del self.group_params.data_sets

        gc.collect()
        print("Loading Saved Clusters")
        self.load_saved_clusters()

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        for i in range(len(train_seq_elements)):
            seq_element = train_seq_elements[i]
            seq_element.cluster_label = self.train_labels[i]
        for i in range(len(test_seq_elements)):
            seq_element = test_seq_elements[i]
            seq_element.cluster_label = self.test_labels[i]

        self.clusters = StockCluster.objects.filter(cluster_group=self)

        for cluster in self.clusters:
            cluster_label = cluster.label
            cur_train_seq_elements = [
                x for x in train_seq_elements if x.cluster_label == cluster_label
            ]
            cur_test_seq_elements = [
                x for x in test_seq_elements if x.cluster_label == cluster_label
            ]
            cluster.initialize(
                cur_train_seq_elements,
                cur_test_seq_elements,
                self.group_params.X_feature_dict,
                self.group_params.y_feature_dict,
            )
        print("Finished Loading Saved Clusters")
    
    def get_cluster(self,cluster_id):
        '''
        Method to get a cluster by its id. We need the cluster to be loaded so we need to 
        avoid querying from the database again as clusters from the database are not fully loaded in
        '''
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster


class Cluster(models.Model):
    """
    Abstract class to encapsulate a cluster. This class is abstract. A cluster is a group of sequences that were assigned the same label in
    a clustering algorithm.

    label: is the label assigned to this cluster
    model_file_string: is the path to the directory where the model is saved
    cluster_metrics: is a dictionary containing metrics for this cluster
    """

    label = models.IntegerField(default=-1)
    cluster_metrics = models.JSONField(default=dict)
    centroid = models.JSONField(default=list)

    class Meta:
        abstract = True

    def initialize(
        self, train_seq_elements, test_seq_elements, X_feature_dict, y_feature_dict
    ):
        """
        Method to initialize the cluster. This method is called after the cluster is created and the sequences are assigned to it.
        """
        self.train_seq_elements = train_seq_elements
        self.test_seq_elements = test_seq_elements
        self.get_3d_array()
        self.group_name = self.cluster_group.group_params.name
        self.X_feature_dict = X_feature_dict
        self.y_feature_dict = y_feature_dict

        self.cluster_dir = f"SavedModels/{self.group_name}/Cluster{self.label}/"

        self.models = RNNModel.objects.filter(cluster=self)
        [
            model.add_elements(self.X_train, self.y_train, self.X_test, self.y_test)
            for model in self.models
        ]

    def get_3d_array(self):
        """
        Method to get the 3d array of sequences from the sequence set
        """
        if len(self.train_seq_elements) == 0 or len(self.test_seq_elements) == 0:
            raise ValueError("No sequences in this cluster")

        self.X_train, self.y_train = SequenceElement.create_array(
            self.train_seq_elements
        )
        self.X_test, self.y_test = SequenceElement.create_array(self.test_seq_elements)

        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def get_model(self, model_id):
        '''
        Method to get a model by its id. We need the model to be loaded so we need to 
        avoid querying from the database again as models from the database are not fully loaded in
        '''
        for model in self.models:
            if model.id == model_id:
                return model


class StockCluster(Cluster):
    """
    Implementation of the Cluster class for stock data
    """

    cluster_group = models.ForeignKey(
        StockClusterGroup, on_delete=models.CASCADE, related_name="clusters_obj"
    )

    def remove_outliers(self):
        pass

    def visualize_cluster(self, isTrain=True, y_range=[-1, 1]):

        if isTrain:
            arr_3d = self.X_train
        else:
            arr_3d = self.X_test

        # get cluster features and corresponding index from X_feature_dict
        cluster_features = self.cluster_group.group_params.cluster_features

        X_cluster = self.cluster_group.filter_by_features(
            arr_3d,
            self.cluster_group.group_params.cluster_features,
            self.X_feature_dict,
        )

        traces = []
        avg_cluster = np.mean(X_cluster, axis=0)
        std_cluster = np.std(X_cluster, axis=0)

        upper_bound = avg_cluster + std_cluster
        lower_bound = avg_cluster - std_cluster

        x = np.arange(avg_cluster.shape[0])[::-1]

        for feature_idx in range(avg_cluster.shape[1]):
            feature = cluster_features[feature_idx]
            text_labels = (
                [""] * (len(x) // 2) + [feature] + [""] * (len(x) - len(x) // 2)
            )
            y_avg = np.ones(avg_cluster.shape[0]) * feature_idx

            z_avg = avg_cluster[::-1, feature_idx]
            z_upper = upper_bound[::-1, feature_idx]
            z_lower = lower_bound[::-1, feature_idx]

            traces.append(
                go.Scatter3d(
                    x=x, y=y_avg, z=z_avg, mode="lines", line=dict(color="red", width=2)
                )
            )
            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y_avg,
                    z=z_upper,
                    line=dict(color="green", width=1),
                    mode="lines+text",
                    text=text_labels,
                )
            )
            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y_avg,
                    z=z_lower,
                    mode="lines",
                    line=dict(color="blue", width=1),
                )
            )

        fig = go.Figure(data=traces)

        camera = dict(
            eye=dict(
                x=1, y=-1.5, z=1.25
            )  # Adjust these values to change the orientation
        )

        fig.update_layout(
            title="Cluster " + str(self.label),
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Feature Index",
                zaxis_title="Value",
                zaxis=dict(range=y_range),
                camera=camera,  # Apply the camera settings
            ),
        )

        return fig
    
    def get_cluster_average(self, isTrain=True):
        if isTrain:
            arr_3d = self.X_train
        else:
            arr_3d = self.X_test

        cluster_features = self.cluster_group.group_params.cluster_features

        X_cluster = self.cluster_group.filter_by_features(
            arr_3d,
            self.cluster_group.group_params.cluster_features,
            self.X_feature_dict,
        )
        avg_cluster = np.mean(X_cluster, axis=0)

        sequence_dict = {}
        for feature_idx in range(avg_cluster.shape[1]):
            feature = cluster_features[feature_idx]
            sequence = avg_cluster[:, feature_idx]
            sequence_dict[feature] = sequence
        
        return sequence_dict

    def visualize_cluster_2d(self, isTrain=True, y_range=[0, 1]):
        if isTrain:
            arr_3d = self.X_train
        else:
            arr_3d = self.X_test

        # get cluster features and corresponding index from X_feature_dict
        cluster_features = self.cluster_group.group_params.cluster_features

        X_cluster = self.cluster_group.filter_by_features(
            arr_3d,
            self.cluster_group.group_params.cluster_features,
            self.X_feature_dict,
        )

        traces = []
        avg_cluster = np.mean(X_cluster, axis=0)

        x = np.arange(avg_cluster.shape[0])

        # manutally create array with 10 colors
        colors = [
            "red",
            "aqua",
            "seagreen",
            "orange",
            "purple",
            "pink",
            "yellow",
            "black",
            "brown",
            "grey",
        ]

        for feature_idx in range(avg_cluster.shape[1]):
            feature = cluster_features[feature_idx]

            z_avg = avg_cluster[:, feature_idx]

            # select random color

            traces.append(
                go.Scatter(
                    x=x,
                    y=z_avg,
                    mode="lines",
                    line=dict(color=colors[feature_idx], width=2),
                    name=feature,
                )
            )

        fig = go.Figure(data=traces)

        fig.update_layout(
            title="Cluster " + str(self.label),
            xaxis_title="Time",
            yaxis_title="Value",
            yaxis=dict(range=y_range),
        )

        return fig

    def train_rnn(
        self,
        model_features,
        target_cols,
        training_dict,
        general_model=None,
        num_feauture_iterations=30,
        sample_size=5,
        ):
        if len(self.X_train) == 0 or len(self.X_test) == 0:
            return

        model_params = self.cluster_group.group_params.model_params
        
        model_params = self.cluster_group.group_params.model_params 

        num_models = 0
        self.models = []

        if not hasattr(self.cluster_group.group_params, 'strong_predictors'):
            self.cluster_group.group_params.strong_predictors = []

        for i in range(num_feauture_iterations):

            target_feature_type = training_dict['target_feature_type']
            max_num_days = training_dict['max_num_days']
            
            if training_dict['random_sample_fut_length']:
                days_to_predict = random.randint(1, max_num_days)
            else:
                days_to_predict = max_num_days

            if target_feature_type == 'lag':
                target_features = ['pctChgclose+{}_target'.format(i) for i in range(1, days_to_predict + 1) ]
            else: 
                target_features = ['sumpctChgclose_{}'.format(i) for i in range(1, days_to_predict + 1) ]
            
            for feature in target_features: 
                if feature not in target_cols: 
                    print(feature)
                    print(target_cols)
                    raise ValueError("Target feature not in target cols")


            y_train_filtered = self.cluster_group.filter_y_by_features(self.y_train, target_features, self.y_feature_dict)
            y_test_filtered = self.cluster_group.filter_y_by_features(self.y_test, target_features, self.y_feature_dict)
            
            features = self.random_sample_features(
                sample_size, model_features, self.cluster_group.group_params.strong_predictors
            )
            print("Features: ", features)
            X_train_filtered = self.cluster_group.filter_by_features(
                self.X_train, features, self.X_feature_dict
            )
            X_test_filtered = self.cluster_group.filter_by_features(
                self.X_test, features, self.X_feature_dict
            )

            for index, model_param in enumerate(model_params):
                if model_param["model_type"] == "SAE":
                    model_type = ModelTypes.SAE
                elif model_param["model_type"] == "AE":
                    model_type = ModelTypes.AE
                elif model_param["model_type"] == "Traditional":
                    model_type = ModelTypes.Traditional
                elif model_param["model_type"] == "AN":
                    model_type = ModelTypes.AN
                else:
                    raise ValueError("Invalid Model Type")

                num_encoder_layers = model_param["num_encoder_layers"]

                model = RNNModel.objects.create(cluster = self, target_feature_type = target_feature_type, target_features = target_features)
                print(
                    "Creating Model "
                    + str(num_models)
                    + " for cluster "
                    + str(self.label)
                    + " model number "
                    + str(index)
                    + " feature iteration "
                    + str(i)
                )
                model.initialize(
                    model_type=model_type,
                    X_train=X_train_filtered,
                    y_train=y_train_filtered,
                    X_test=X_test_filtered,
                    y_test=y_test_filtered,
                    model_features=features,
                    model_dir=self.cluster_dir + "model" + str(num_models) + "/",
                    num_autoencoder_layers=num_encoder_layers,
                    num_encoder_layers=num_encoder_layers,
                )

                num_nn_layers = 0
                for layer in model_param["layers"]:
                    if layer["type"] == "LSTM":
                        if num_nn_layers + 1 == num_encoder_layers:
                            model.addLSTMLayer(
                                layer["units"],
                                return_sequences=False,
                                activation=layer["activation"],
                            )
                        else:
                            model.addLSTMLayer(
                                layer["units"], activation=layer["activation"]
                            )
                        num_nn_layers += 1
                    elif layer["type"] == "GRU":
                        if num_nn_layers + 1 == num_encoder_layers:
                            model.addGRULayer(
                                layer["units"],
                                return_sequences=False,
                                activation=layer["activation"],
                            )
                        else:
                            model.addGRULayer(
                                layer["units"], activation=layer["activation"]
                            )
                        num_nn_layers += 1
                    elif layer["type"] == "Dropout":
                        model.addDropoutLayer(layer["rate"])
                    else:
                        raise ValueError("Invalid Layer Type")

                model.buildModel()
                model.fit(epochs=100, batch_size=20)
                model.evaluate_test()
                model.compute_average_accuracy()
                model.serialize()
                model.save()
                num_models += 1
                self.models.append(model)
                clear_session()

    def get_best_model(self):
        """
        Method to get the best model for this cluster. Currently, the best model is the one with the highest average accuracy
        """
        self.best_model_idx = 0
        best_accuracy = 0
        for i in range(len(self.models)):
            accuracy_list = []
            for step_result in StepResult.objects.filter(RNNModel=self.models[i]):
                accuracy_list.append(step_result.dir_accuracy)

            accuracy = np.mean(accuracy_list)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model_idx = i

        return self.best_model_idx

    def hard_filter_models(self, accuracy_threshold=50, epoch_threshold=5):
        """
        Method to filter the models based on a dictionary of filters
        """

        for model in self.models:
            accuracy = model.model_metrics["avg_accuracy"]
            epochs = model.model_metrics["effective_epochs"]

            print(
                "Cluster "
                + str(self.label)
                + " Model "
                + str(model.id)
                + " Accuracy: "
                + str(accuracy)
                + " Epochs: "
                + str(epochs)
                + "\n"
            )

            if accuracy < accuracy_threshold or epochs < epoch_threshold:
                model.delete()

    def sort_models(self):
        """
        Method to sort the models by accuracy
        """
        self.sorted_models = sorted(
            self.models, key=lambda x: x.compute_average_accuracy(), reverse=True
        )

        return self.sorted_models

    def random_sample_features(
        self, num_features, features, strong_predictors, strong_ratio=0.5
    ):
        """
        Method to randomly sample features from a list of features
        """
        print(num_features)
        num_strong_predictors = math.ceil(num_features * strong_ratio)
        
        training_features = []

        if num_strong_predictors > len(strong_predictors):
            num_strong_predictors = len(strong_predictors)

        

        if num_strong_predictors > 0:
            strong_predictors = np.random.choice(
                strong_predictors, num_strong_predictors, replace=False
            )
            training_features += list(strong_predictors)
        else:
            strong_predictors = []

        num_other_predictors = num_features - len(training_features)
            

        if 'rsi' in features:
            #todo currently hardcoding momentum indicators to make sure they are sampled in larger feature set 
            num_momentum_indicators = 3 
            momentum_indicators = ["rsi", "macd", "macd_signal", "macd_diff", "stoch_k", "stoch_d"]
            training_features += list(np.random.choice(momentum_indicators, num_momentum_indicators, replace=False))
            num_other_predictors -= num_momentum_indicators

        other_predictors = np.random.choice(
            [x for x in features if x not in training_features],
            num_other_predictors,
            replace=False,
        )

        print(len(other_predictors))

        training_features += list(other_predictors)

        return training_features    


@receiver(models.signals.post_delete, sender=StockClusterGroupParams)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """
    Deletes file from filesystem
    when corresponding `StockClusterGroupParams` object is deleted.
    """
    directory = f"SavedModels/{instance.name}/"
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            print(f"Directory '{directory}' and all its contents have been removed.")
        except OSError as error:
            print(f"Error: {error}")


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
        if (
            reinitialize
            and hasattr(layer, "kernel_initializer")
            and hasattr(layer, "bias_initializer")
        ):
            # Get initializers
            kernel_initializer = layer.kernel_initializer
            bias_initializer = layer.bias_initializer

            # Reinitialize weights
            if hasattr(layer, "kernel"):
                layer.kernel.assign(kernel_initializer(shape=layer.kernel.shape))
            if hasattr(layer, "bias"):
                layer.bias.assign(bias_initializer(shape=layer.bias.shape))

    new_model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mae")

    return new_model


def create_attention(input_shape, output_steps, num_features):
    # Encoder
    encoder_inputs = Input(shape=(None, input_shape))
    encoder_lstm1 = LSTM(units=25, return_sequences=True, activation="tanh")(
        encoder_inputs
    )
    encoder_lstm2, state_h, state_c = LSTM(
        units=50, return_sequences=True, return_state=True, activation="tanh"
    )(encoder_lstm1)

    # Attention mechanism
    attention = tf.keras.layers.Attention()
    # Decoder
    # Initialize LSTM decoder layer
    decoder_lstm = LSTM(
        units=50, return_sequences=False, return_state=True, activation="tanh"
    )

    all_decoder_outputs = []
    decoder_input = tf.zeros_like(
        encoder_inputs[:, 0, :]
    )  # Initial decoder input, zeros
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
        context_vector, attention_weights = attention(
            [query, encoder_lstm2], return_attention_scores=True
        )

        # Pass the context vector and previous states to LSTM decoder
        decoder_output, state_h, state_c = decoder_lstm(
            context_vector, initial_state=decoder_states
        )

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
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    return model
