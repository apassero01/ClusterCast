import pandas as pd 
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from itertools import accumulate
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import ClusterPipeline.models.SequencePreprocessing as sp
import tensorflow as tf
from tqdm.keras import TqdmCallback


def filter_by_features(seq, feature_list, X_feature_dict):
    """
    Method to filter a 3d array of sequences by a list of features.
    """
    indices = [X_feature_dict[x] for x in feature_list]
    # Using numpy's advanced indexing to select the required features
    return seq[:, :, indices]

def filter_y_by_features(seq, feature_list, y_feature_dict):
    '''
    Method to filter a 3d array of sequences by a list of features.
    '''
    indices = [y_feature_dict[x] for x in feature_list]
    # Using numpy's advanced indexing to select the required features
    return seq[:, indices]



class Cluster: 
    def __init__(self, label, X_train, y_train, X_test, y_test, train_seq_elements, test_seq_elements):
        self.label = label
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_set_length = len(X_train)
        self.test_set_length = len(X_test)
        self.train_seq_elements = train_seq_elements
        self.test_seq_elements = test_seq_elements

        self.current_pred_index = 0

    
    def train_model(self, model, epochs, batch_size, shuffle = False, patience = 10):
        self.model = model
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        print(self.X_train.shape, self.y_train.shape)

        self.model.fit(self.X_train, self.y_train, epochs=epochs, shuffle = shuffle, batch_size=batch_size, validation_data=(self.X_test, self.y_test), callbacks=[early_stopping], verbose = 1)
        return self.model

    


    def eval_model(self, test_model = True):

        self.model_result_dict = {}

        model_output = self.model.predict(self.X_test)
        print(type(model_output), len(model_output))

        if test_model:
            self.predicted_y = model_output[0]
            self.attention_weights = model_output[1]
        else:
            self.predicted_y = model_output
            self.attention_weights = None
        
        self.predicted_y = np.squeeze(self.predicted_y, axis=-1)

        print(self.y_test.shape)


        
        self.predicted_y_cum = np.cumsum(self.predicted_y, axis=1)
        self.y_test_cum = np.cumsum(self.y_test, axis=1)
        
    

        num_days = self.predicted_y.shape[1]  # Assuming this is the number of days
        print(num_days)
        results = pd.DataFrame(self.predicted_y_cum, columns=[f'predicted_{i+1}' for i in range(num_days)])

        for i in range(num_days):
            results[f'real_{i+1}'] = self.y_test_cum[:, i]

        # Generate output string with accuracies
        output_string = f"Cluster Number:\n"
        for i in range(num_days):
            tolerance = 0.05  # Set your tolerance level

            # Modify the condition for 'same_day'
            results['same_day'] = ((results[f'predicted_{i+1}'] > 0) & (results[f'real_{i+1}'] > 0)) | \
                        ((results[f'predicted_{i+1}'] < 0) & (results[f'real_{i+1}'] < 0)) | \
                        (np.abs(results[f'predicted_{i+1}'] - results[f'real_{i+1}']) < tolerance)
            accuracy = round(results['same_day'].mean() * 100,2)

            self.model_result_dict[str(i+1)] = {}
            self.model_result_dict[str(i+1)]['accuracy'] = accuracy
            self.model_result_dict[str(i+1)]['total_predictions'] = len(results['same_day'])
            self.model_result_dict[str(i+1)]['correct_predictions'] = results['same_day'].sum()
            self.model_result_dict[str(i+1)]['mse'] = np.mean((results[f'predicted_{i+1}'] - results[f'real_{i+1}'])**2)
            self.model_result_dict[str(i+1)]['mae'] = np.mean(np.abs(results[f'predicted_{i+1}'] - results[f'real_{i+1}']))
            cum_predicted = results[[f'predicted_{j+1}' for j in range(i+1)]].cumsum(axis=1).iloc[:, -1]
            cum_real = results[[f'real_{j+1}' for j in range(i+1)]].cumsum(axis=1).iloc[:, -1]
            # self.model_result_dict[str(i+1)]['dtw'] = test_utils_models.dtw_loss()

            mae = np.mean(np.abs(cum_predicted - cum_real))
            mse = np.mean((cum_predicted - cum_real)**2)

            self.model_result_dict[str(i+1)]['cum_mse'] = mse
            self.model_result_dict[str(i+1)]['cum_mae'] = mae

            output_string += (
                f"Accuracy{i+1}D {accuracy}% "
                f"PredictedRet: {results[f'predicted_{i+1}'].mean()} "
                f"ActRet: {results[f'real_{i+1}'].mean()}\n"
            )
        
        output_string += f"Train set length:  Test set length: {len(self.y_test_cum)}\n"

        self.output_string = output_string

        self.results = results

        return output_string, results, self.attention_weights
    
    def eval_model_ragged(self, test_model=True):
        self.model_result_dict = {}

        model_output = self.model.predict(self.X_test)
        print(type(model_output), len(model_output))

        if test_model:
            self.predicted_y = model_output[0]
            self.attention_weights = model_output[1]
        else:
            self.predicted_y = model_output
            self.attention_weights = None
        
        self.predicted_y = np.squeeze(self.predicted_y, axis=-1)
        self.y_test = np.squeeze(self.y_test, axis=-1)

        self.predicted_y_modified, self.y_test_modified = truncate_sequences(self.predicted_y, self.y_test)
        
        self.predicted_y_cum = np.cumsum(self.predicted_y_modified, axis=1)
        self.y_test_cum = np.cumsum(self.y_test_modified, axis=1)
        
    

        num_days = self.predicted_y.shape[1]  # Assuming this is the number of days
        print(num_days)
        results = pd.DataFrame(self.predicted_y_cum, columns=[f'predicted_{i+1}' for i in range(num_days)])

        for i in range(num_days):
            results[f'real_{i+1}'] = self.y_test_cum[:, i]

        # Generate output string with accuracies
        output_string = f"Cluster Number:\n"
        for i in range(num_days):
            tolerance = 0.05  # Set your tolerance level

            # Modify the condition for 'same_day'
            results['same_day'] = ((results[f'predicted_{i+1}'] > 0) & (results[f'real_{i+1}'] > 0)) | \
                        ((results[f'predicted_{i+1}'] < 0) & (results[f'real_{i+1}'] < 0)) | \
                        (np.abs(results[f'predicted_{i+1}'] - results[f'real_{i+1}']) < tolerance)
            accuracy = round(results['same_day'].mean() * 100,2)

            self.model_result_dict[str(i+1)] = {}
            self.model_result_dict[str(i+1)]['accuracy'] = accuracy
            self.model_result_dict[str(i+1)]['total_predictions'] = len(results['same_day'])
            self.model_result_dict[str(i+1)]['correct_predictions'] = results['same_day'].sum()
            self.model_result_dict[str(i+1)]['mse'] = np.mean((results[f'predicted_{i+1}'] - results[f'real_{i+1}'])**2)
            self.model_result_dict[str(i+1)]['mae'] = np.mean(np.abs(results[f'predicted_{i+1}'] - results[f'real_{i+1}']))

            output_string += (
                f"Accuracy{i+1}D {accuracy}% "
                f"PredictedRet: {results[f'predicted_{i+1}'].mean()} "
                f"ActRet: {results[f'real_{i+1}'].mean()}\n"
            )
        
        output_string += f"Train set length:  Test set length: {len(self.y_test_cum)}\n"

        self.results = results

        return output_string, results, self.attention_weights
    
    def eval_model_prob(self,y_test, test_model = True):

        self.model_result_dict = {}

        model_output = self.model.predict(self.X_test)
        print(type(model_output), len(model_output))

        if test_model:
            output = model_output[0]
            self.attention_weights = model_output[1]
        else:
            output = model_output
            self.attention_weights = None

        mean, variance = tf.split(output, 2, axis=-1)

        self.predicted_y = mean
        self.predicted_y = self.predicted_y.numpy().reshape(-1, self.predicted_y.shape[1])

        print(y_test)

    
        
        self.predicted_y_cum = np.cumsum(self.predicted_y, axis=1)
        self.y_test_cum = np.cumsum(y_test, axis=1)
        
    

        num_days = self.predicted_y.shape[1]  # Assuming this is the number of days
        print(num_days)
        results = pd.DataFrame(self.predicted_y_cum, columns=[f'predicted_{i+1}' for i in range(num_days)])

        for i in range(num_days):
            results[f'real_{i+1}'] = self.y_test_cum[:, i]

        # Generate output string with accuracies
        output_string = f"Cluster Number:\n"
        for i in range(num_days):
            tolerance = 0.05  # Set your tolerance level

            # Modify the condition for 'same_day'
            results['same_day'] = ((results[f'predicted_{i+1}'] > 0) & (results[f'real_{i+1}'] > 0)) | \
                        ((results[f'predicted_{i+1}'] < 0) & (results[f'real_{i+1}'] < 0)) | \
                        (np.abs(results[f'predicted_{i+1}'] - results[f'real_{i+1}']) < tolerance)
            accuracy = round(results['same_day'].mean() * 100,2)

            self.model_result_dict[str(i+1)] = {}
            self.model_result_dict[str(i+1)]['accuracy'] = accuracy
            self.model_result_dict[str(i+1)]['total_predictions'] = len(results['same_day'])
            self.model_result_dict[str(i+1)]['correct_predictions'] = results['same_day'].sum()
            self.model_result_dict[str(i+1)]['mse'] = np.mean((results[f'predicted_{i+1}'] - results[f'real_{i+1}'])**2)
            self.model_result_dict[str(i+1)]['mae'] = np.mean(np.abs(results[f'predicted_{i+1}'] - results[f'real_{i+1}']))

            output_string += (
                f"Accuracy{i+1}D {accuracy}% "
                f"PredictedRet: {results[f'predicted_{i+1}'].mean()} "
                f"ActRet: {results[f'real_{i+1}'].mean()}\n"
            )
        
        output_string += f"Train set length:  Test set length: {len(self.y_test_cum)}\n"

        self.results = results

        return output_string, results, self.attention_weights

    def inverse_transform(self,  y_feature_sets, target_feature ):
        
        self.predicted_y_transformed = np.zeros_like(self.predicted_y)
        for i, feature in enumerate(target_feature):
            scaler = [feature_set for feature_set in y_feature_sets if feature == feature_set.name][0].scaler
            self.predicted_y_transformed[:, i] = scaler.inverse_transform(self.predicted_y[:, i].reshape(-1, 1)).reshape(-1)
    
    def get_next_prediction_value(self, seq_start_date):
        """
        Method to get the next prediction value
        """
        # find seq element in test_seq element with start date = seq_start_date and return index of it 
        index = [i for i, elem in enumerate(self.test_seq_elements) if elem.start_date == seq_start_date][0]
        
        return self.predicted_y_transformed[index], self.attention_weights[index]


def visualize_future_movement_mpl(sequence_element, prediction_change_1, prediction_change_2, scaler, isCuma = False, target_features = None, num_days = 6):
    """
    Visualizes the sequence
    """
    # target_cols = target_feature_set.cols 
    actual_values_change = sp.SequenceElement.filter_y_by_features(sequence_element.seq_y, target_features, sequence_element.y_feature_dict)
    print(actual_values_change.shape)
    actual_values_change = scaler.inverse_transform(actual_values_change)



    if not isCuma: 
        prediction_change_1 = prediction_change_1[-num_days:]
        prediction_change_2 = prediction_change_2[-num_days:]
        actual_values_change = actual_values_change[-num_days:]
        prediction_change_1 = list(accumulate(prediction_change_1))
        prediction_change_2 = list(accumulate(prediction_change_2))
        actual_values_change = list(accumulate(actual_values_change))


    ohlc_data = sp.SequenceElement.filter_by_features(sequence_element.seq_x, ['open', 'high', 'low', 'close', 'volume', 'ema50'], sequence_element.x_feature_dict)
    data = {
        'Open': ohlc_data[:, 0],
        'High': ohlc_data[:, 1],
        'Low': ohlc_data[:, 2],
        'Close': ohlc_data[:, 3],
        'Volume': ohlc_data[:, 4],
        'ema50': ohlc_data[:, 5]
    }

    holiday_calendar = USFederalHolidayCalendar()
    holidays = holiday_calendar.holidays()
    market_calendar = CustomBusinessDay(calendar=holiday_calendar)

    start_date = sequence_element.start_date
    end_date = sequence_element.end_date
    historical_dates = pd.date_range(start=start_date, end=end_date, freq=market_calendar)

    if len(historical_dates) != len(data['Close']):
        if len(historical_dates) > len(data['Close']):
            historical_dates = historical_dates[:len(data['Close'])]
        else:
            extra_dates = pd.date_range(start=historical_dates[-1], periods=len(data['Close']) - len(historical_dates)+1, freq=market_calendar)[1:]
            print(extra_dates)
            historical_dates = np.concatenate([historical_dates, extra_dates])
    historical_df = pd.DataFrame(data, index=historical_dates)

    last_close = historical_df['Close'][-5:].mean()



    actual_values = [last_close + last_close*x/100 for x in actual_values_change]
    prediction_1 = [last_close + last_close*x/100 for x in prediction_change_1]
    prediction_2 = [last_close + last_close*x/100 for x in prediction_change_2]

    temp_future_dates = pd.date_range(start=historical_dates[-1], periods=len(actual_values) + 1, freq=market_calendar)

    # If the first date in temp_future_dates is the same as the last date in historical_dates, exclude it
    if temp_future_dates[0] == historical_dates[-1]:
        future_dates = temp_future_dates[1:]
    else:
        future_dates = temp_future_dates[:-1]

    actual_series = pd.Series(data=actual_values, index=future_dates)
    prediction_series_1 = pd.Series(data=prediction_1, index=future_dates)
    prediction_series_2 = pd.Series(data=prediction_2, index=future_dates)

    extended_actual_values = np.concatenate([np.full(len(historical_dates), np.nan), actual_series])
    extended_prediction_values_1 = np.concatenate([np.full(len(historical_dates), np.nan), prediction_series_1])
    extended_prediction_values_2 = np.concatenate([np.full(len(historical_dates), np.nan), prediction_series_2])


    actual_plot = mpf.make_addplot(extended_actual_values, type='line', markersize=50, marker='o', 
                                color='green', linestyle='-', label='Actual')

    prediction_plot_1= mpf.make_addplot(extended_prediction_values_1, type='line', markersize=50, marker='o', 
                                    color='red', linestyle='-', label='Baseline_predction')
    
    prediction_plot_2 = mpf.make_addplot(extended_prediction_values_2, type='line', markersize=50, marker='o',
                                    color='blue', linestyle='-', label='cluster_prediction')
    

    # Define the mplfinance style
    style = mpf.make_mpf_style(base_mpf_style='default', rc={'figure.figsize': (8, 4)})

    combined_df = pd.concat([historical_df, pd.DataFrame(index=future_dates)], sort=False)

    ema_plot = mpf.make_addplot(combined_df['ema50'], color='blue', label='EMA50')

    # Plot the data
    mpf.plot(combined_df, type='candle', style=style, addplot=[actual_plot, prediction_plot_1,prediction_plot_2, ema_plot],
            figscale=1.5, volume=True)
    

def visualize_future_movement_mpl_close(sequence_element, predictions, actuals, price_scaler, trend_scaler):
    """
    Visualizes the sequence with actual close prices and multiple predictions.
    
    Args:
    - sequence_element: Contains sequence data including start and end dates.
    - predictions: List of arrays, each with shape (798, 15, 1) representing different prediction sets.
    - actuals: Actual closing prices corresponding to the predictions.
    """
    # Extract OHLC and other data
    ohlc_data = sp.SequenceElement.filter_by_features(sequence_element.seq_x, ['open', 'high', 'low', 'close', 'volume', 'ema50'], sequence_element.x_feature_dict)
    data = {
        'Open': ohlc_data[:, 0],
        'High': ohlc_data[:, 1],
        'Low': ohlc_data[:, 2],
        'Close': ohlc_data[:, 3],
        'Volume': ohlc_data[:, 4],
        'ema50': ohlc_data[:, 5]
    }
    
    # Prepare date index for historical data
    market_calendar = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    start_date = sequence_element.start_date
    end_date = sequence_element.end_date

    historical_dates = pd.date_range(start=start_date, end=end_date, freq=market_calendar)

    if len(historical_dates) != len(data['Close']):
        if len(historical_dates) > len(data['Close']):
            historical_dates = historical_dates[:len(data['Close'])]
        else:
            extra_dates = pd.date_range(start=historical_dates[-1], periods=len(data['Close']) - len(historical_dates)+1, freq=market_calendar)[1:]
            print(extra_dates)
            historical_dates = np.concatenate([historical_dates, extra_dates])
    historical_df = pd.DataFrame(data, index=historical_dates)

    # historical_df[['Open', 'High', 'Low', 'Close', 'ema50']] = price_scaler.inverse_transform(historical_df[['Open', 'High', 'Low', 'Close', 'ema50']])
    # historical_df['Volume'] = trend_scaler.inverse_transform(historical_df[['Volume']])

    # Prepare future dates for predictions and actuals
    num_future_days = predictions[0].shape[0]  # Assuming all prediction arrays have the same second dimension
    future_dates = pd.date_range(start=historical_dates[-1], periods=num_future_days + 1, freq=market_calendar)[1:]  # Exclude overlapping start date

    # Create series for predictions and actuals
    prediction_series = [pd.Series(data=prediction.squeeze(), index=future_dates) for prediction in predictions]
    actual_series = pd.Series(data=actuals.squeeze(), index=future_dates)

    # Concatenate historical and future data
    extended_df = pd.concat([historical_df, pd.DataFrame(index=future_dates)], sort=False)
    extended_df.loc[future_dates, 'Actuals'] = actual_series
    for i, series in enumerate(prediction_series):
        extended_df.loc[future_dates, f'Prediction_{i+1}'] = series

    # Plotting
    apds = [mpf.make_addplot(extended_df['Actuals'], type='line', color='green', linestyle='-', marker='o', markersize=5, label='Actual')]
    for i in range(len(prediction_series)):
        apds.append(mpf.make_addplot(extended_df[f'Prediction_{i+1}'], type='line', color=['red', 'blue', 'orange'][i % 3], linestyle='-', marker='x', markersize=5, label=f'Prediction {i+1}'))

    # Define the mplfinance style
    style = mpf.make_mpf_style(base_mpf_style='default', rc={'figure.figsize': (8, 4)})
    mpf.plot(extended_df, type='candle', style=style, addplot=apds, volume=True, title='Stock Price Prediction Visualization')

    


def plot_results(baseline_dict, cluster_dict):
    """
    Method to plot the results of the baseline and clusters
    """
    # Plotting the baseline
    baseline_results = pd.DataFrame(baseline_dict)
    baseline_results = baseline_results.T
    baseline_results['model'] = 'baseline'

    # Plotting the cluster results
    cluster_results = pd.DataFrame(cluster_dict)
    cluster_results = cluster_results.T
    cluster_results['model'] = 'cluster'

    # Concatenating the results
    results = pd.concat([baseline_results, cluster_results])

    # make line plot 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=results, x=results.index.astype(int), y='accuracy', hue='model', ax=ax)
    ax.set_title('Accuracy of Baseline and Cluster Models')
    ax.set_xlabel('Days')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(title='Model')
    fig.show()

    # line plot of mse 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=results, x=results.index.astype(int), y='mse', hue='model', ax=ax)
    ax.set_title('Mean Squared Error of Baseline and Cluster Models')
    ax.set_xlabel('Days')
    ax.set_ylabel('MSE')
    ax.legend(title='Model')
    fig.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=results, x=results.index.astype(int), y='mae', hue='model', ax=ax)
    ax.set_title('Mean Absolute Error of Baseline and Cluster Models')
    ax.set_xlabel('Days')
    ax.set_ylabel('MAE')
    ax.legend(title='Model')
    fig.show()

    # line plot of cum_mse
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=results, x=results.index.astype(int), y='cum_mse', hue='model', ax=ax)
    ax.set_title('Cumulative Mean Squared Error of Baseline and Cluster Models')
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative MSE')
    ax.legend(title='Model')
    fig.show()

    # line plot of cum_mae
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=results, x=results.index.astype(int), y='cum_mae', hue='model', ax=ax)
    ax.set_title('Cumulative Mean Absolute Error of Baseline and Cluster Models')
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative MAE')
    ax.legend(title='Model')
    fig.show()
    


def adjust_sequence(seq, max_len, end_token=-999):
    """
    Truncate a sequence randomly and then pad it to a specified maximum length.
    """
    # Randomly choose a truncation point for the sequence, ensuring at least 1 element is kept
    truncation_point = np.random.randint(1, len(seq) + 1)
    # Truncate the sequence to the random truncation point
    truncated_seq = seq[:truncation_point]
    # Pad the truncated sequence to the specified max_len with the end_token
    return np.pad(truncated_seq, (0, max_len - len(truncated_seq)), 'constant', constant_values=end_token)

def augment_dataset(X, y, max_steps):
    """
    Augment a dataset by truncating and padding the output sequences to a fixed length.
    """
    y_adjusted = np.array([adjust_sequence(y[i], max_steps) for i in range(len(y))])
    return X, y_adjusted

def augment_data(X_train, y_train, X_test, y_test, max_steps):
    """
    Augment training and testing datasets by truncating and padding output sequences to a fixed length.
    """
    X_train_augmented, y_train_augmented = augment_dataset(X_train, y_train, max_steps)
    X_test_augmented, y_test_augmented = augment_dataset(X_test, y_test, max_steps)

    return X_train_augmented, y_train_augmented, X_test_augmented, y_test_augmented

def append_end_token(y_train, y_test, end_token=-999):
    """
    Appends an end token to each sequence in the training and testing datasets.

    Parameters:
    - y_train (numpy array): Training target dataset with shape (batch_size, output_steps).
    - y_test (numpy array): Testing target dataset with shape (batch_size, output_steps).
    - end_token (int, optional): The value to append as an end token. Default is -999.

    Returns:
    - Tuple of numpy arrays: (y_train_new, y_test_new) with the new shape (batch_size, output_steps + 1).
    """
    
    # Append the end_token to each sequence in y_train and y_test
    y_train_new = np.hstack((y_train, np.full((y_train.shape[0], 1), end_token)))
    y_test_new = np.hstack((y_test, np.full((y_test.shape[0], 1), end_token)))

    return y_train_new, y_test_new

def augment_and_sample(X, y, max_steps, iterations, sample_size):
    """
    Augment data and sample from it over a number of iterations.

    Parameters:
    - X (np.array): Input features.
    - y (np.array): Output targets.
    - max_steps (int): Maximum sequence length.
    - iterations (int): Number of augmentation iterations to perform.
    - sample_size (int): Number of samples to take from each augmented set.

    Returns:
    - np.array: Aggregated samples from augmented data.
    """
    # Initialize arrays to hold the cumulative sampled data
    X_sampled = np.empty((0, X.shape[1], X.shape[2]))  # Adjust the second dimension based on your feature set
    y_sampled = np.empty((0, max_steps))

    for _ in range(iterations):
        # Augment the data
        _, y_augmented = augment_dataset(X, y, max_steps)

        # Randomly sample from the augmented data
        indices = np.random.choice(y_augmented.shape[0], size=min(sample_size, y_augmented.shape[0]), replace=False)
        X_sampled_iteration = X[indices]
        y_sampled_iteration = y_augmented[indices]

        # Append the new samples to the cumulative dataset
        X_sampled = np.vstack((X_sampled, X_sampled_iteration))
        y_sampled = np.vstack((y_sampled, y_sampled_iteration))

    return X_sampled, y_sampled


def truncate_sequences(predicted_y, y_test):
    # Ensure arrays are of float type to accommodate NaNs
    predicted_y = predicted_y.astype(float)
    y_test = y_test.astype(float)
    
    # Step 1: Identify where -999 appears in either array
    pred_end_mask = (predicted_y == -999)
    test_end_mask = (y_test == -999)
    
    # Step 2: Determine the first index of -999 in either array for each sequence
    pred_end_index = np.where(pred_end_mask.any(axis=1), pred_end_mask.argmax(axis=1), predicted_y.shape[1])
    test_end_index = np.where(test_end_mask.any(axis=1), test_end_mask.argmax(axis=1), y_test.shape[1])

    # Step 3: Determine the minimum end index for each sequence
    min_end_index = np.minimum(pred_end_index, test_end_index)
    
    # Step 4: Apply NaN masking beyond the first -999 occurrence for each sequence
    for i in range(predicted_y.shape[0]):
        # Only modify if -999 was found and there are values beyond the index to modify
        if min_end_index[i] < predicted_y.shape[1]:
            predicted_y[i, min_end_index[i]:] = np.nan
            y_test[i, min_end_index[i]:] = np.nan

    return predicted_y, y_test