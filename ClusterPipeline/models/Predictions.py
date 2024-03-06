from django.db import models
from .ClusterProcessing import StockClusterGroup, Cluster, StockClusterGroupParams
from .TSeriesPreproccesing import StockDataSet
from .SequencePreprocessing import ScalingMethod, SequenceElement
from .SequencePreprocessing import StockSequenceSet
from tensorflow.keras.backend import clear_session
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from django.core.files.base import ContentFile
from .RNNModels import RNNModel, StepResult, ModelPrediction
from collections import defaultdict
import tensorflow as tf
from datetime import date
import shutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from django.db.models.signals import post_delete
from django.dispatch import receiver
from itertools import accumulate
import gc


def prediction_directory_path(instance, filename):
    """
    Returns the path of the directory where the file will be saved.
    Function is passed as a parameter when declaring FileField in model.
    """
    date_str = instance.prediction_start_date.strftime("%Y-%m-%d %H:%M:%S")
    return os.path.join(
        "predictions", f"{instance.ticker}-{instance.interval}-{date_str}", filename
    )


class Prediction(models.Model):
    pass


class StockPrediction(Prediction):
    """
    Class Responsible for making predictions on a stock.
    It is uniquly identified by the ticker, interval, and prediction_start_date where prediction_start_date is the date of the first prediction.
    The process for making a prediction starts with finding all ClusterGroups that predict on the ticker and interval specified.
    Then, for each ClusterGroup, we find the matching clusters and models that have an accuracy above the threshold specified.
    Finally, we make predictions on the matching models and save the predictions in a DataFrame.
    """

    ticker = models.CharField(max_length=10)
    interval = models.CharField(max_length=10)
    prediction_start_date = models.DateTimeField()
    df_file = models.FileField(
        upload_to=prediction_directory_path, blank=True, null=True
    )
    forcast_timeline = models.ForeignKey(
        "StockForcastTimeline", on_delete=models.CASCADE, blank=True, null=True
    )
    dir_path = models.CharField(max_length=100, blank=True, null=True)
    final_prediction_date = models.DateTimeField(blank=True, null=True)
    cluster_group_params = models.JSONField(default=list, blank=True, null=True)

    def initialize(self):
        """
        Initialize a calandar of business days and ensure the prediction_start_date is indeed a trading day
        """
        holiday_calendar = USFederalHolidayCalendar()
        holidays = holiday_calendar.holidays()
        self.market_calendar = CustomBusinessDay(calendar=holiday_calendar)

        self.dir_path = os.path.join(
            "media",
            "predictions",
            f'{self.ticker}-{self.interval}-{self.prediction_start_date.strftime("%Y-%m-%d %H:%M:%S")}',
        )

    def predict_all_groups(
        self,
        length_factor=0.25,
        total_model_accuracy_thresh=77,
        individual_model_accuracy_thresh=65,
        epochs_threshold=10,
    ):
        """
        Predicts on all groups that predict on the ticker and interval specified.
        This method iterates through all groups and calls lower level methods (predict_by_group) on each group.
        It formats all of the output dataframes into a merged dataframe and saves the dataframe to disk.
        """
        self.create_general_data_set("2020-01-01", self.prediction_start_date)
        all_group_params = StockClusterGroupParams.objects.filter(
            interval=self.interval
        )
        all_group_params = [
            group_params
            for group_params in all_group_params
            if self.ticker in group_params.tickers
        ]
        print([group_params.id for group_params in all_group_params])
        all_groups = [
            StockClusterGroup.objects.get(group_params=group_params)
            for group_params in all_group_params
            if group_params.id not in self.cluster_group_params
        ]
        all_rnn_predictions = []
        for cluster_group in all_groups:
            try:
                cluster_group.load_saved_group()
            except Exception as e:
                print(e)
                print(cluster_group.id)
                continue
            print("Predicting on group {}".format(cluster_group.id))
            rnn_predictions = self.predict_by_group(
                cluster_group,
                length_factor,
                total_model_accuracy_thresh,
                individual_model_accuracy_thresh,
                epochs_threshold,
            )
            all_rnn_predictions.append(rnn_predictions)
            self.cluster_group_params.append(cluster_group.group_params.id)

            for rnn_prediction in rnn_predictions:
                if (
                    self.final_prediction_date is None
                    or rnn_prediction.end_date > self.final_prediction_date
                ):
                    self.final_prediction_date = rnn_prediction.end_date
            del cluster_group
            gc.collect()

        return all_rnn_predictions

    def predict_by_group(
        self,
        cluster_group,
        length_factor=0.1,
        total_model_accuracy_thresh=90,
        individual_model_accuracy_thresh=77,
        epochs_threshold=20,
    ):
        """
        Method to predict on all models associated with a single cluster_group.

        The Process Goes as follows:
        1. We take the generic dataset mirror it (@mirror_group()) to create a dataset that matches the group_params of the cluster_group
        2. We iterate through all future_sequence_elements in the sequence set returned by mirror_group()
            a. A future_sequence_element is a sequence element that contains NaN values for the target variables (has predictive power into the future)
        3. For each future_sequence_element, we find the matching clusters in the cluster_group
            a. A matching cluster is a cluster that has a centroid that is close to the future_sequence_element
        4. For each matching cluster, we find the models that have an accuracy above the threshold specified
        5. For each model, we make a prediction on the future_sequence_element and save the prediction in a DataFrame
        6. We merge all of the DataFrames into a single DataFrame and return it

        """
        sequence_set = self.mirror_group(self.generic_dataset, cluster_group)
        future_sequence_elements = sequence_set.group_params.future_seq_elements


        rnn_predictions = []

        future_sequence_elements = [
            future_sequence_elements[0]
        ]  # Dont think we need to go through them all as that has been doing on previous days

        # find the matching clusters
        model_dict = defaultdict(list)

        for future_seq_element in future_sequence_elements:
            matching_clusters, X_cluster = self.find_matching_clusters(
                future_seq_element,
                cluster_group,
                sequence_set.group_params.X_feature_dict,
                length_factor,
            )

            # print num matching cluster and cluster id
            print(
                "Number of matching clusters: {} and {} for group {}".format(
                    len(matching_clusters),
                    [cluster.label for cluster in matching_clusters],
                    cluster_group.id,
                )
            )

            for cluster in matching_clusters:
                models = RNNModel.objects.filter(cluster=cluster)
                for model in models:
                    avg_accuracy = model.model_metrics["avg_accuracy"]
                    epochs = model.model_metrics["effective_epochs"]
                    # print("Model {} has accuracy {} and effective epochs {}".format(model.id, avg_accuracy, epochs))
                    if (
                        avg_accuracy > total_model_accuracy_thresh
                        and epochs > epochs_threshold
                    ):
                        model_dict[model.id].append(future_seq_element)
                        print("ADDED MODEL")

                if len(model_dict) > 0:
                    print("VISUALIZING CLUSTER")
                    print(X_cluster.shape)
                    combined_fig = self.visualize_current_and_cluster(
                        cluster, X_cluster, cluster_group.group_params.cluster_features
                    )
                    self.save_cluster_visualization(combined_fig, cluster.id)

        for model_id in model_dict.keys():
            num_preds = 0
            model = RNNModel.objects.get(id=model_id)

            seq_elements = model_dict[model_id]
            # create 2d array of sequences to predict on
            input_data = []
            for seq_element in seq_elements:
                pred_seq = self.filter_by_features(
                    seq_element.seq_x_scaled,
                    model.model_features,
                    sequence_set.group_params.X_feature_dict,
                )
                input_data.append(pred_seq)

            input_data = np.array(input_data)

            model.deserialize_model()

            predictions = model.predict(input_data)
            predictions = predictions.squeeze(axis=2)

            for i in range(len(seq_elements)):
                end_date = seq_elements[i].end_date
                cur_prediction = predictions[i]
                future_dates = pd.date_range(
                    end_date, periods=len(cur_prediction) + 1, freq=self.market_calendar
                )[1:].tolist()

                cur_prediction_transformed = np.zeros_like(cur_prediction)
                for i,feature in enumerate(cluster_group.group_params.target_cols):
                    scaler = [feature_set for feature_set in cluster_group.group_params.y_feature_sets if feature_set.name == feature][0].scaler
                    cur_prediction_transformed[i] = scaler.inverse_transform(cur_prediction[i].reshape(-1,1)).squeeze()


                end_day_close = self.generic_dataset.test_df.loc[end_date]["close"]
                print("PRESCALED PREDICTION")
                print(cur_prediction)


                if model.target_feature_type == "lag":
                    if len(cur_prediction == 15):
                        cur_prediction_transformed = list(accumulate(cur_prediction_transformed[-15:]))
                    else:
                        cur_prediction_transformed = list(accumulate(cur_prediction_transformed[-6:]))

                price_predictions = []
                daily_accuracy = []
                step_results = StepResult.objects.filter(RNNModel=model)
                for pred, step_result in zip(cur_prediction_transformed, step_results):
                    if step_result.dir_accuracy < individual_model_accuracy_thresh:
                        price_predictions.append(None)
                        continue

                    daily_accuracy.append(step_result.dir_accuracy)
                    predicted_price = pred / 100 * end_day_close + end_day_close
                    price_predictions.append(round(predicted_price, 2))

                daily_accuracy = [acc for acc in daily_accuracy if not np.isnan(acc)]
                if len(daily_accuracy) == 0:
                    continue

                print("Daily Accuracy")
                print(daily_accuracy)
                formatted_dates = [
                    date.strftime("%Y-%m-%d %H:%M:%S") for date in future_dates
                ]

                rnn_prediction = model.create_prediction(
                    price_predictions, formatted_dates, self, end_day_close
                )
                rnn_predictions.append(rnn_prediction)

                num_preds += 1

            # clear session and delete model
            clear_session()
            del model.model

        return rnn_predictions

    def mirror_group(self, stock_dataset, cluster_group):
        """
        Creates a dataset that mirrors the group_params specified.
        This dataset will be used to predict on the group_params dataset.
        1. Scales the data in dataframe format
        2. Create sequences of specified length
        3. Scales seqeuences with features specified as ScalingMethod.SBS or SBSG
        4. Creates cuma_pctChg features

        Returns a StockSequenceSet object that contains the mirrored dataset

        Note: It would be ideal to have a params object that contains all of the parameters necessary to mirror a group and
        a separate object that contains the rest of the parameters. Currently all of these params live in StockClusterGroupParams

        It is not worth the time to make this change now, So this function selects all parameters that are absolutely necessary to mirror the group
        from StockClusterGroupParsm and copies them to the StockDataSet object.

        Necessary For Mirroring:
        1. n_steps: new group must have same step length
        2. cluster_features: new group must cluster using the same features
        3. training_features: new group must train using the same features
        4. X_feature_sets: new group must have the same X_feature_sets which contain the scalers needed to transform the new dataset
        5. y_feature_sets: new group must have the same y_feature_sets which contain the scalers needed to transform the new dataset
        6. scaling_dict: new group must have the same scaling_dict which contains the scaling methods for each feature type
        """

        # copy the generic dataset
        mirrored_dataset = copy.deepcopy(stock_dataset)

        # Extract the group params from the group object we are trying to mirror
        db_group_params = copy.deepcopy(cluster_group.group_params)

        # Extract the parameters from the db_group_params that are necessary to mirror the group
        n_steps = db_group_params.n_steps
        cluster_features = db_group_params.cluster_features
        training_features = db_group_params.training_features
        X_feature_sets = [
            feature_set
            for feature_set in db_group_params.X_feature_sets
            if feature_set.ticker == self.ticker
        ]
        y_feature_sets = [
            feature_set
            for feature_set in db_group_params.y_feature_sets
            if feature_set.ticker == self.ticker
        ]
        scaling_dict = db_group_params.scaling_dict

        # Update the saved group params to fit the current situation
        mirrored_dataset.group_params.n_steps = n_steps
        mirrored_dataset.group_params.cluster_features = cluster_features
        mirrored_dataset.group_params.training_features = training_features
        mirrored_dataset.group_params.X_feature_sets = X_feature_sets
        mirrored_dataset.group_params.y_feature_sets = y_feature_sets
        mirrored_dataset.X_feature_sets = X_feature_sets
        mirrored_dataset.y_feature_sets = y_feature_sets
        mirrored_dataset.group_params.scaling_dict = scaling_dict
        mirrored_dataset.group_params.data_sets = [mirrored_dataset]

        # scale the data
        quant_min_max_feature_sets = [
            feature_set
            for feature_set in X_feature_sets
            if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value
            or feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value
        ]
        quant_min_max_feature_sets += [
            feature_set
            for feature_set in y_feature_sets
            if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value
            or feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value
        ]

        mirrored_dataset.test_df = mirrored_dataset.scale_transform(
            mirrored_dataset.test_df, quant_min_max_feature_sets
        )

        sequence_set = StockSequenceSet(mirrored_dataset.group_params)
        sequence_set.create_combined_sequence()
        sequence_set.scale_sequences()
        sequence_set.create_cuma_pctChg_features()

        return sequence_set

    def create_general_data_set(self, start_date, end_date):
        """
        Creates a stock dataset for the ticker and interval specified.
        This prescaled dataset will be used for all group configurations and will contain all features
        """

        # temporarily, we have the target features defined as follows
        target_features = [
            "sumpctChgclose_1",
            "sumpctChgclose_2",
            "sumpctChgclose_3",
            "sumpctChgclose_4",
            "sumpctChgclose_5",
            "sumpctChgclose_6",
        ]

        scaling_dict = {
            "price_vars": ScalingMethod.UNSCALED,
            "trend_vars": ScalingMethod.UNSCALED,
            "pctChg_vars": ScalingMethod.STANDARD,
            "rolling_vars": ScalingMethod.STANDARD,
            "target_vars": ScalingMethod.QUANT_MINMAX,
            "lag_feature_vars": ScalingMethod.STANDARD,
            "momentum_vars": ScalingMethod.STANDARD,
        }
        group_params = StockClusterGroupParams(
            tickers=[self.ticker],
            interval=self.interval,
            start_date=start_date,
            end_date=self.prediction_start_date,
            target_cols=target_features,
            cluster_features=[],
        )
        group_params.initialize()
        group_params.scaling_dict = scaling_dict
        group_params.initialize()

        stock_dataset = StockDataSet(group_params, self.ticker)

        stock_dataset.create_dataset()
        stock_dataset.create_features()
        stock_dataset.create_y_targets(group_params.target_cols)
        stock_dataset.train_test_split(training_percentage=0)

        self.generic_dataset = stock_dataset

    def find_matching_clusters(
        self, future_sequence_element, cluster_group, feature_dict, length_factor=0.1
    ):
        """
        Returns a list of clusters that the current stock_sequence_set matches with respect to cluster_features
        """
        seq_x = future_sequence_element.seq_x_scaled

        cluster_seq_x = self.filter_by_features(
            seq_x, cluster_group.group_params.cluster_features, feature_dict
        )
        smallest_distance = np.inf
        smallest_cluster = None
        for cluster in cluster_group.clusters:
            distance = self.compute_distance(cluster_seq_x, cluster.centroid)
            if distance < smallest_distance:
                smallest_distance = distance
                smallest_cluster = cluster

        return [smallest_cluster], cluster_seq_x

        ## Go back and possibly implemenet distance based matching alg
        # matching_clusters = []
        # for cluster in cluster_group.clusters:
        #     cluster_metrics = cluster.cluster_metrics
        #     centroid = cluster.centroid
        #     std = cluster_metrics['std_dev']
        #     iqr = cluster_metrics['iqr']

        #     distance = self.compute_distance(cluster_seq_x, centroid)

        #     threshold = std * (1 + length_factor * future_sequence_element.seq_x_scaled.shape[0])

        #     print("Distance: {}".format(distance))
        #     print("Threshold: {}".format(threshold))

        #     if distance < threshold:
        #         matching_clusters.append(cluster)

        return matching_clusters

    def compute_distance(self, future_sequence, centroid, metric="euclidean"):
        """
        Computes the distance between the sequence_element and the centroid
        """
        if metric == "euclidean":
            return np.linalg.norm(future_sequence - centroid)
        elif metric == "dtw":
            return dtw(future_sequence, centroid)
        else:
            raise Exception("Metric not supported")

    def filter_by_features(self, seq, features, feature_dict):
        """
        Returns a sequence that only contains the features specified
        """
        indices = [feature_dict[feature] for feature in features]

        return seq[:, indices]

    def visualize_current_and_cluster(self, cluster, X_cluster, cluster_features):
        # Create a subplot with 1 row and 2 columns
        subplots_fig = make_subplots(
            rows=1, cols=2, subplot_titles=("avg_cluster", "current cluster")
        )

        # Get the cluster visualization
        cluster_fig = cluster.visualize_cluster_2d()

        # Add traces from cluster_fig to the first subplot
        for trace in cluster_fig.data:
            subplots_fig.add_trace(trace, row=1, col=1)

        x = np.arange(X_cluster.shape[0])
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

        # Add traces for the current cluster to the second subplot
        for feature_idx in range(X_cluster.shape[1]):
            feature = cluster_features[feature_idx]
            y = X_cluster[:, feature_idx]
            subplots_fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=feature + "_cur",
                    line=dict(color=colors[feature_idx]),
                ),
                row=1,
                col=2,
            )

        # Update layout for the entire subplot figure
        subplots_fig.update_layout(
            title="Cluster "
            + str(cluster.id)
            + " Group: "
            + str(cluster.cluster_group.id),
            xaxis_title="Time",
            yaxis_title="Value",
            yaxis=dict(range=[0, 1]),
        )

        return subplots_fig

    def save_cluster_visualization(self, fig, cluster_id):
        """
        Saves the plotly figure to disk
        """
        file_path = os.path.join(self.dir_path, f"clusters/cluster-{cluster_id}.html")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        fig.write_html(file_path)

    def load_cluster_visualization(self, cluster_id):
        """
        Loads the plotly figure from disk
        """
        file_path = os.path.join(self.dir_path, f"clusters/cluster-{cluster_id}.html")
        fig = go.Figure()
        fig = fig.from_html(file_path)
        return fig

    def create_prediction_output(self):
        """
        Generate the json object from all model predictions for the forcast timeline frontend
        """

        model_predictions = list(
            self.stock_model_predictions.all().order_by("start_date")
        )

        dates = pd.date_range(
            self.prediction_start_date,
            self.final_prediction_date,
            freq=self.market_calendar,
        ).tolist()
        dates = [date.strftime("%Y-%m-%d %H:%M:%S") for date in dates]

        model_prediction_output = []
        for model_prediction in model_predictions:
            model_prediction_output.append(model_prediction.create_model_pred_dict())

        return dates, model_prediction_output

    def rebuild_predictions(self, model_prediction_output):
        for pred_output in model_prediction_output:
            model_prediction = ModelPrediction.objects.get(
                pk=pred_output["model_prediction_id"]
            )
            model_prediction.update_prediction(pred_output)
            model_prediction.save()


class ForcastTimeline(models.Model):
    pass


class StockForcastTimeline(ForcastTimeline):
    """
    Class to encpasualte the dynamic, running timeline of predictions for a specific stock and interval
    This contains the aggreated @StockPrediction objects for a specific stock and interval.
    """

    ticker = models.CharField(max_length=10)
    interval = models.CharField(max_length=10)
    prediction_start_date = models.DateTimeField()
    prediction_end_date = models.DateTimeField()
    final_prediction_date = models.DateTimeField(blank=True, null=True)

    def initialize(self):
        """
        Initialize a calandar of business days and ensure the prediction_start_date is indeed a trading day
        """
        holiday_calendar = USFederalHolidayCalendar()
        holidays = holiday_calendar.holidays()
        self.market_calendar = CustomBusinessDay(calendar=holiday_calendar)

        if self.prediction_start_date in holidays:
            next_business_day = self.market_calendar.rollforward(
                self.prediction_start_date
            )
            self.prediction_start_date = next_business_day

        self.stock_predictions = list(
            StockPrediction.objects.filter(forcast_timeline=self)
        )
        # sort stock predictions by prediction_start_date

        if len(self.stock_predictions) > 0:
            self.stock_predictions = sorted(
                self.stock_predictions, key=lambda x: x.prediction_start_date
            )

            self.prediction_dates = []
            for stock_prediction in self.stock_predictions:
                self.prediction_dates.append(stock_prediction.prediction_start_date)

            self.prediction_end_date = self.prediction_dates[-1]
            self.prediction_start_date = self.prediction_dates[0]
            self.final_prediction_date = self.stock_predictions[
                -1
            ].final_prediction_date
        else:
            self.prediction_dates = []

        self.save()

    def add_prediction_range(
        self,
        start_date,
        end_date,
        overwrite=False,
        total_model_accuracy_thresh=90,
        individual_model_accuracy_thresh=68,
        epochs_threshold=10,
    ):
        """
        Creates a range of dates to predict on
        """

        prediction_range = pd.date_range(
            start_date, end_date, freq=self.market_calendar
        ).tolist()

        current_predictions = []

        for date in prediction_range:
            if date in self.prediction_dates and not overwrite:
                continue
            print("Predicting on date {}".format(date))
            stock_prediction = StockPrediction(
                ticker=self.ticker,
                interval=self.interval,
                prediction_start_date=date,
                forcast_timeline=self,
            )
            stock_prediction.save()
            stock_prediction.initialize()
            stock_prediction.predict_all_groups(
                total_model_accuracy_thresh=total_model_accuracy_thresh,
                individual_model_accuracy_thresh=individual_model_accuracy_thresh,
                epochs_threshold=epochs_threshold,
            )
            if stock_prediction.final_prediction_date is None:
                stock_prediction.delete()
                continue

            self.prediction_dates.append(date)
            stock_prediction.save()
            self.stock_predictions.append(stock_prediction)
            current_predictions.append(stock_prediction)

        # sort date list and prediction list
        if len(self.stock_predictions) > 0:
            self.prediction_dates, self.stock_predictions = zip(
                *sorted(zip(self.prediction_dates, self.stock_predictions))
            )
            self.prediction_start_date = self.prediction_dates[0]
            self.prediction_end_date = self.prediction_dates[-1]

            for i in range(len(self.stock_predictions) - 1, 0, -1):
                if self.stock_predictions[i].final_prediction_date is not None:
                    self.final_prediction_date = self.stock_predictions[
                        i
                    ].final_prediction_date
                    break

    def create_prediction_output(self, prediction_start_date, prediction_end_date):
        """
        Generate the json object from all model predictions for the forcast timeline frontend
        """

        if len(self.stock_predictions) == 0 or self.final_prediction_date is None:
            return [], []

        model_predictions_output = []

        print(self.market_calendar)
        dates = pd.date_range(
            start=self.prediction_start_date,
            end=self.final_prediction_date,
            freq=self.market_calendar,
        ).tolist()
        dates = [date.strftime("%Y-%m-%d %H:%M:%S") for date in dates]

        for stock_prediction in self.stock_predictions:
            if (
                stock_prediction.prediction_start_date < prediction_start_date
                or stock_prediction.prediction_start_date > prediction_end_date
            ):
                continue
            stock_prediction.initialize()
            (
                cur_dates,
                model_prediction_output,
            ) = stock_prediction.create_prediction_output()
            model_predictions_output.append(
                {
                    "prediction_id": stock_prediction.id,
                    "prediction_start_date": stock_prediction.prediction_start_date.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "prediction_end_date": stock_prediction.final_prediction_date.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "results": model_prediction_output,
                }
            )

        return dates, model_predictions_output

    def rebuild_predictions(self, model_predictions_output):
        for pred_output in model_predictions_output:
            stock_prediction = StockPrediction.objects.get(
                pk=pred_output["prediction_id"]
            )
            stock_prediction.rebuild_predictions(pred_output["results"])
            stock_prediction.save()


@receiver(post_delete, sender=StockPrediction)
def submission_delete(sender, instance, **kwargs):
    """
    Deletes the directory containing the prediction files if prediction is deleted
    """

    directory_path = os.path.join(
        "media",
        "predictions",
        f"{instance.ticker}-{instance.interval}-{instance.prediction_start_date}",
    )
    print("Deleting everything in {}".format(directory_path))
    # Check if the directory exists
    if os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(
                f"Directory '{directory_path}' and all its contents have been removed."
            )
        except OSError as error:
            print(f"Error: {error}")


@receiver(post_delete, sender=StockForcastTimeline)
def submission_delete(sender, instance, **kwargs):
    """
    Deletes the directory containing the prediction files if prediction is deleted
    """
    # Assuming 'prediction_directory_path' is a function that returns the path of the directory.
    directory_path = os.path.join(
        "media",
        "forcast_timelines",
        f"{instance.ticker}-{instance.interval}",
    )
    print("Deleting everything in {}".format(directory_path))
    # Check if the directory exists
    if os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(
                f"Directory '{directory_path}' and all its contents have been removed."
            )
        except OSError as error:
            print(f"Error: {error}")
