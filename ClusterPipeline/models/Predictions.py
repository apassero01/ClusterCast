from django.db import models
from .ClusterProcessing import StockClusterGroup, Cluster, StockClusterGroupParams
from .TSeriesPreproccesing import StockDataSet
from .SequencePreprocessing import ScalingMethod, SequenceElement
from .SequencePreprocessing import StockSequenceSet
from tensorflow.keras.backend import clear_session
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from django.core.files.base import ContentFile
from .RNNModels import RNNModel
from collections import defaultdict
import tensorflow as tf
from datetime import date
import shutil
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import copy 
import pandas as pd
import numpy as np
from tslearn.metrics import dtw 
from django.db.models.signals import post_delete
from django.dispatch import receiver

def prediction_directory_path(instance, filename):
    '''
    Returns the path of the directory where the file will be saved. 
    Function is passed as a parameter when declaring FileField in model. 
    '''
    date_str = instance.prediction_start_date
    return os.path.join('predictions', f'{instance.ticker}-{instance.interval}-{date_str}', filename)  

class Prediction(models.Model): 
    pass

class StockPrediction(Prediction):
    '''
    Class Responsible for making predictions on a stock.
    It is uniquly identified by the ticker, interval, and prediction_start_date where prediction_start_date is the date of the first prediction.
    The process for making a prediction starts with finding all ClusterGroups that predict on the ticker and interval specified.
    Then, for each ClusterGroup, we find the matching clusters and models that have an accuracy above the threshold specified.
    Finally, we make predictions on the matching models and save the predictions in a DataFrame.
    '''
    ticker = models.CharField(max_length=10)
    interval = models.CharField(max_length=10)
    prediction_start_date = models.DateField()
    df_file = models.FileField(upload_to=prediction_directory_path, blank=True, null=True)
    
    def initialize(self):
        '''
        Initialize a calandar of business days and ensure the prediction_start_date is indeed a trading day
        '''
        holiday_calendar = USFederalHolidayCalendar()
        holidays = holiday_calendar.holidays()
        self.market_calendar = CustomBusinessDay(calendar=holiday_calendar)

        if self.prediction_start_date in holidays: 
            next_business_day = self.market_calendar.rollforward(self.prediction_start_date)
            self.prediction_start_date = next_business_day
        
        self.df_file = models.FileField(upload_to=prediction_directory_path, blank=True, null=True)

        

    def predict_all_groups(self, length_factor = .25,model_accuracy_threshold = 77,epochs_threshold = 10): 
        '''
        Predicts on all groups that predict on the ticker and interval specified.
        This method iterates through all groups and calls lower level methods (predict_by_group) on each group.
        It formats all of the output dataframes into a merged dataframe and saves the dataframe to disk.
        '''
        self.create_general_data_set('2020-01-01', self.prediction_start_date)
        all_group_params = StockClusterGroupParams.objects.filter(interval = self.interval)
        all_group_params = [group_params for group_params in all_group_params if self.ticker in group_params.tickers]
        all_groups = [StockClusterGroup.objects.get(group_params = group_params) for group_params in all_group_params]
        all_dfs = [] 
        for cluster_group in all_groups:
            try:
                cluster_group.load_saved_group()
            except Exception as e:
                print(e)
                print(cluster_group.id)
                continue

            df = self.predict_by_group(cluster_group, length_factor, model_accuracy_threshold, epochs_threshold)
            all_dfs.append(df.T)
        
        joined_df = self.create_pred_df(all_dfs)
        self.save_data_frame(joined_df, all_groups)

        return joined_df


    def predict_by_group(self, cluster_group, length_factor = .1,model_accuracy_threshold = 90,epochs_threshold = 20): 
        '''
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

        '''
        sequence_set = self.mirror_group(self.generic_dataset, cluster_group)
        future_sequence_elements = sequence_set.group_params.future_seq_elements

        target_scaler = sequence_set.group_params.y_feature_sets[0].scaler

        prediction_dfs = [] 

        # find the matching clusters
        model_dict = defaultdict(list)

        for future_seq_element in future_sequence_elements:
            matching_clusters = self.find_matching_clusters(future_seq_element, cluster_group, sequence_set.group_params.X_feature_dict, length_factor)

            # print num matching cluster and cluster id 
            print("Number of matching clusters: {} and {} for group {}".format(len(matching_clusters), [cluster.label for cluster in matching_clusters], cluster_group.id))
    
            for cluster in matching_clusters:
                models = RNNModel.objects.filter(cluster = cluster)
                for model in models:
                    avg_accuracy = model.model_metrics["avg_accuracy"]
                    epochs = model.model_metrics["effective_epochs"]
                    # print("Model {} has accuracy {} and effective epochs {}".format(model.id, avg_accuracy, epochs))
                    if avg_accuracy > model_accuracy_threshold and epochs > epochs_threshold:
                        model_dict[model.id].append(future_seq_element)
                        print("ADDED MODEL")

        for model_id in model_dict.keys():
            num_preds = 0
            model = RNNModel.objects.get(id = model_id)

            seq_elements = model_dict[model_id]
            # create 2d array of sequences to predict on
            input_data = [] 
            for seq_element in seq_elements:
                pred_seq = self.filter_by_features(seq_element.seq_x_scaled, model.model_features, sequence_set.group_params.X_feature_dict)
                input_data.append(pred_seq)
            
            input_data = np.array(input_data)

            model.deserialize_model()

            predictions = model.predict(input_data)
            predictions = predictions.squeeze(axis=2)

            for i in range(len(seq_elements)):
                end_date = seq_elements[i].end_date
                cur_prediction = predictions[i]
                future_dates = pd.date_range(end_date, periods = len(cur_prediction)+1, freq = self.market_calendar)[1:].tolist()
                
                end_day_close = self.generic_dataset.test_df.loc[end_date]['close']
                cur_prediction = target_scaler.inverse_transform(cur_prediction)

                price_predictions = []
                for pred in cur_prediction:
                    predicted_price = pred/100 * end_day_close + end_day_close
                    price_predictions.append(round(predicted_price,2))
            
                formatted_dates = [date.strftime('%Y-%m-%d') for date in future_dates]
                rows = {
                    'date': ['avg_accuracy', 'effective_epochs'] + formatted_dates,
                    str(model_id) + '-' + str(num_preds): [model.model_metrics['avg_accuracy'], model.model_metrics['effective_epochs']] + price_predictions
                }

                # Convert rows to DataFrame
                pred_df = pd.DataFrame(rows)
                prediction_dfs.append(pred_df)
                num_preds += 1

            # clear session and delete model 
            clear_session()
            del model.model
        
        joined_df = self.create_pred_df(prediction_dfs)
        self.save_data_frame(joined_df, [cluster_group])

        return joined_df

    def mirror_group(self,stock_dataset, cluster_group):
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
        X_feature_sets = [feature_set for feature_set in db_group_params.X_feature_sets if feature_set.ticker == self.ticker]
        y_feature_sets = [feature_set for feature_set in db_group_params.y_feature_sets if feature_set.ticker == self.ticker]
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
        quant_min_max_feature_sets = [feature_set for feature_set in X_feature_sets if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value or feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value]
        quant_min_max_feature_sets += [feature_set for feature_set in y_feature_sets if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value or feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value]

        mirrored_dataset.test_df = mirrored_dataset.scale_transform(mirrored_dataset.test_df, quant_min_max_feature_sets)

        sequence_set = StockSequenceSet(mirrored_dataset.group_params)
        sequence_set.create_combined_sequence()
        sequence_set.scale_sequences()
        sequence_set.create_cuma_pctChg_features()

        return sequence_set

    def create_general_data_set(self,start_date,end_date):
        """
        Creates a stock dataset for the ticker and interval specified. 
        This prescaled dataset will be used for all group configurations and will contain all features
        """

        # temporarily, we have the target features defined as follows 
        target_features = ['sumpctChgclose_1','sumpctChgclose_2','sumpctChgclose_3','sumpctChgclose_4','sumpctChgclose_5','sumpctChgclose_6']

        scaling_dict = {
            'price_vars': ScalingMethod.UNSCALED,
            'trend_vars' : ScalingMethod.SBS,
            'pctChg_vars' : ScalingMethod.QUANT_MINMAX,
            'rolling_vars' : ScalingMethod.QUANT_MINMAX,
            'target_vars' : ScalingMethod.QUANT_MINMAX
        }
        group_params = StockClusterGroupParams(tickers = [self.ticker], interval = self.interval, start_date = start_date, end_date = end_date, target_cols = target_features, cluster_features = [])
        group_params.scaling_dict = scaling_dict

        stock_dataset = StockDataSet(group_params,self.ticker)

        stock_dataset.create_dataset()
        stock_dataset.create_features()
        stock_dataset.create_y_targets(group_params.target_cols)
        stock_dataset.train_test_split(training_percentage=0)

        self.generic_dataset = stock_dataset
    

    def visualize_future_predictions(self, df_predictions):
        '''
        Creates a plotly figure that visualizes the closing prices and predictions for the ticker and interval specified.
        Currently, as we wanted dynamic visualization, we converted this function to js and moved it to the frontend.
        Method is not currently used but still good to have
        '''
        # Assuming self.generic_dataset.test_df is a DataFrame available in the class
        close_prices = self.generic_dataset.test_df.tail(20)['close']

        closing_dates = close_prices.index

        # Create a figure
        fig = go.Figure()

        # Add trace for closing prices
        fig.add_trace(go.Scatter(x=closing_dates, y=close_prices, mode='lines+markers', name='Closing Prices'))

        # Process predictions
        date_columns = [col for col in df_predictions.columns if col.startswith('202') or col.startswith('203')]
        prediction_dates = pd.to_datetime(date_columns, format='%Y-%m-%d')
        prediction_dates = prediction_dates[prediction_dates > closing_dates[-1]]
        prediction_values = df_predictions.loc[:, prediction_dates.strftime('%Y-%m-%d')]

        median_predictions = prediction_values.median(axis=0)
        iq1_predictions = prediction_values.quantile(0.25, axis=0,numeric_only = False)
        iq3_predictions = prediction_values.quantile(0.75, axis=0,numeric_only = False)

        last_price = close_prices.iloc[-1]
        percent_change = (median_predictions - last_price) / last_price * 100

        # Add trace for predictions
        fig.add_trace(go.Scatter(x=prediction_dates, y=median_predictions, mode='lines+markers', name='Predictions', line=dict(color='red')))

        # Add IQR area
        fig.add_trace(go.Scatter(x=prediction_dates, y=iq1_predictions, fill=None, mode='lines', line=dict(color='lightgreen'), showlegend=False))
        fig.add_trace(go.Scatter(x=prediction_dates, y=iq3_predictions, fill='tonexty', mode='lines', line=dict(color='lightgreen'), name='Predictions IQR'))

        # Adding text labels for percent changes
        for i, date in enumerate(prediction_dates):
            fig.add_annotation(x=date, y=median_predictions[i], text=f'{percent_change.iloc[i]:.2f}%', showarrow=False, yshift=10)

        closing_dates_str = closing_dates.strftime('%Y-%m-%d').tolist()
        prediction_dates_str = prediction_dates.strftime('%Y-%m-%d').tolist()
        all_dates_str = closing_dates_str + prediction_dates_str

        fig.update_layout(
            title=f'Closing and Predicted Prices for {self.ticker}',
            xaxis=dict(
                title='Date',
                type='category',  # This specifies that xaxis is categorical
                tickmode='array',
                tickvals=list(range(len(all_dates_str))),  # Position for each tick
                ticktext=all_dates_str,  # Date string for each tick
                tickangle=-90
            ),
            yaxis_title='Price',
            legend=dict(y=1, x=1),
            hovermode='x'
        )

        return fig

    def find_matching_clusters(self, future_sequence_element, cluster_group, feature_dict, length_factor = .1):
        """
        Returns a list of clusters that the current stock_sequence_set matches with respect to cluster_features
        """
        seq_x = future_sequence_element.seq_x_scaled
        
        cluster_seq_x = self.filter_by_features(seq_x, cluster_group.group_params.cluster_features, feature_dict)
        smallest_distance = np.inf
        smallest_cluster = None
        for cluster in cluster_group.clusters: 
            distance = self.compute_distance(cluster_seq_x, cluster.centroid)
            if distance < smallest_distance:
                smallest_distance = distance
                smallest_cluster = cluster
        
        return [smallest_cluster]

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

    def compute_distance(self, future_sequence, centroid, metric = 'euclidean'):
        """
        Computes the distance between the sequence_element and the centroid
        """
        if metric == 'euclidean':
            return np.linalg.norm(future_sequence - centroid)
        elif metric == 'dtw':
            return dtw(future_sequence, centroid)
        else:
            raise Exception("Metric not supported")
    
    def filter_by_features(self, seq, features, feature_dict): 
        """
        Returns a sequence that only contains the features specified
        """
        indices = [feature_dict[feature] for feature in features]

        return seq[:,indices]
    

    def create_pred_df(self, prediction_list):
        '''
        Merge all dfs in the list using outer join on date
        '''
        concat_df = pd.DataFrame({'date': []})

        if len(prediction_list) > 0:
            for df in prediction_list:
                if df.empty:
                    continue
                concat_df = concat_df.merge(df, how='outer', on='date')

        # Transpose the DataFrame so that dates become the row indices
        concat_df = concat_df.set_index('date').T


        if concat_df.empty:
            return concat_df

        date_columns = [col for col in concat_df.columns if col.startswith('202') or col.startswith('203')]
        non_date_columns = ['avg_accuracy', 'effective_epochs']
        
        # Sort by the datetime index
        date_columns_sorted = sorted(date_columns)

        # Reorder the DataFrame columns by placing the date columns first (sorted), followed by the non-date columns
        final_columns_order = date_columns_sorted + non_date_columns
        concat_df_sorted = concat_df[final_columns_order]

        return concat_df_sorted
    
    def save_data_frame(self, df, cluster_groups):
        '''
        Saves the dataframe to disk
        '''
        # Fill NaN values with a placeholder
        df_filled = df.fillna('NaN_placeholder')  # Replace 'NaN_placeholder' with your chosen placeholder
        json_data = df_filled.to_json(orient='split')
        
        group_ids = [cluster_group.id for cluster_group in cluster_groups]
        file_name = 'pred_df-' + str(group_ids) + '.json'

        self.df_file = ContentFile(json_data, file_name)
    
    def load_data_frame(self):
        '''
        Loads the dataframe from disk
        '''
        with open(self.df_file.path, 'r') as file:
            json_data = file.read()
        
        # Load DataFrame with the same orientation
        df = pd.read_json(json_data, orient='split')
        
        # Convert placeholder back to NaN
        df.replace('NaN_placeholder', pd.NA, inplace=True)
        return df


@receiver(post_delete, sender=StockPrediction)
def submission_delete(sender, instance, **kwargs):
    '''
    Deletes the directory containing the prediction files if prediction is deleted
    '''
    # Assuming 'prediction_directory_path' is a function that returns the path of the directory.
    directory_path = os.path.join('media','predictions', f'{instance.ticker}-{instance.interval}-{instance.prediction_start_date}')
    print("Deleting everything in {}".format(directory_path))
    # Check if the directory exists
    if os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' and all its contents have been removed.")
        except OSError as error:
            print(f"Error: {error}")
