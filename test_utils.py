import os
from copy import deepcopy

os.environ["DJANGO_SETTINGS_MODULE"] = "ClusterCast.settings"
import django

django.setup()
import ClusterPipeline.models.ClusterProcessing as cp
import ClusterPipeline.models.SequencePreprocessing as sp
from asgiref.sync import sync_to_async
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from copy import deepcopy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    TimeDistributed,
    Dense,
    Concatenate,
    Permute,
    Reshape,
    Multiply,
    RepeatVector,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras.backend as K
from keras.layers import Layer
from keras.layers import Activation, Flatten
from tensorflow.keras.regularizers import L1, L2, L1L2
import tensorflow as tf
import datetime
from importlib import reload


@sync_to_async
def extract_cluster_groups(group_ids):
    groups = []
    for id in group_ids:
        group = cp.StockClusterGroup.objects.get(pk=id)
        group.load_saved_group()
        groups.append(group)
    return groups


@sync_to_async
def get_all_clusters(groups):
    clusters = []
    for group in groups:
        clusters += group.clusters

    return clusters

@sync_to_async
def recreate_group(group):
    group_params = group.group_params
    group_params.pk = None
    
    group = cp.StockClusterGroup.objects.create(group_params=deepcopy(group.group_params))
    group.generate_new_group()

def visualize_cluster_centroid(centroid, cluster_features):
    # Extract the number of features
    num_features = centroid.shape[1]

    # Set up the time steps
    time_steps = range(centroid.shape[0])

    # Plot the centroid for each feature
    for i in range(num_features):
        plt.scatter(time_steps, centroid[:, i], label=f"Feature {cluster_features[i]}")
        plt.plot(
            time_steps, centroid[:, i], "kx-"
        )  # 'kx-' for black crosses with lines

    plt.xlabel("Time Steps")
    plt.ylabel("Feature Values")
    plt.legend()
    plt.title("Centroid Visualization Over Time Steps")

    return plt


def cluster_a_group(group, train_tuple, n_clusters):
    X_train, y_train, X_test, y_test = deepcopy(train_tuple)

    feature_dict = group.group_params.X_feature_dict
    cluster_features = group.group_params.cluster_features
    X_train_cluster = group.filter_by_features(X_train, cluster_features, feature_dict)
    X_test_cluster = group.filter_by_features(X_test, cluster_features, feature_dict)

    alg = TimeSeriesKMeans(
        n_clusters=n_clusters, metric="euclidean", max_iter=5, random_state=0
    )

    alg.fit(X_train_cluster)

    labels = alg.labels_
    test_labels = alg.predict(X_test_cluster)

    cluster_list = []

    for i in range(n_clusters):
        X_train_new = deepcopy(X_train[labels == i])
        y_train_new = deepcopy(y_train[labels == i])
        X_test_new = deepcopy(X_test[test_labels == i])
        y_test_new = deepcopy(y_test[test_labels == i])
        cluster_list.append(
            (
                alg.cluster_centers_[i],
                (X_train_new, y_train_new, X_test_new, y_test_new),
            )
        )

    return cluster_list


def create_regular_model(input_shape, latent_dim=6):
    # Input layer
    input_layer = Input(shape=(None, input_shape))

    # masking_layer = Masking(mask_value=0.0, name='masking_layer')(input_layer)

    # Encoder

    encoder_lstm2 = LSTM(
        units=100,
        activation="tanh",
        return_sequences=True,
        name="encoder_lstm_2_restore",
    )(input_layer)
    encoder_dropout2 = Dropout(0.2, name="encoder_dropout_2_restore")(encoder_lstm2)

    encoder_lstm3 = LSTM(
        units=50,
        activation="tanh",
        return_sequences=False,
        name="encoder_lstm_3_restore",
    )(encoder_dropout2)
    encoder_dropout3 = Dropout(0.2, name="encoder_dropout_3_restore")(encoder_lstm3)

    # encoder_lstm4 = LSTM(units=50, activation='tanh', return_sequences=False, name='encoder_lstm_4_restore')(encoder_dropout3)
    # encoder_dropout4 = Dropout(0.2, name='encoder_dropout_4_restore')(encoder_lstm4)

    # Repeat Vector
    repeat_vector = RepeatVector(latent_dim, name="repeat_vector")(encoder_dropout3)

    # Decoder
    decoder_lstm1 = LSTM(
        units=100,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm_1_restore",
    )(repeat_vector)
    decoder_dropout1 = Dropout(0.2, name="decoder_dropout_1_restore")(decoder_lstm1)

    decoder_lstm2 = LSTM(
        units=50,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm_2_restore",
    )(decoder_dropout1)
    decoder_dropout2 = Dropout(0.2, name="decoder_dropout_2_restore")(decoder_lstm2)

    time_distributed_output = TimeDistributed(Dense(1), name="time_distributed_output")(
        decoder_dropout2
    )

    # Create the model
    model_lstm = Model(inputs=input_layer, outputs=time_distributed_output)

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model_lstm.compile(optimizer=optimizer, loss="mae")

    return model_lstm


def custom_profit_loss_percent_change(y_true, y_pred):
    """
    Custom loss function to maximize profit based on predicted percent changes.
    
    Parameters:
    - y_true: Actual percent changes. Expected shape (batch_size, sequence_length).
    - y_pred: Predicted percent changes. Expected shape (batch_size, sequence_length).
    
    Returns:
    - A scalar loss value to be minimized.
    """
    # Determine the positions: 1 for buy (positive prediction), -1 for sell (negative prediction)
    positions = tf.where(y_pred > 0, 1.0, -1.0)
    
    # Calculate profits: product of actual percent changes and positions
    profits = positions * y_true

    loss = -tf.reduce_sum(profits)
    
    # To maximize profit, we minimize the negative sum of profits
    penalty = tf.reduce_mean(tf.square(y_pred - y_true))
    
    # Combine the loss and the penalty
    total_loss = loss + penalty
    
    return total_loss


def attention_mechanism(encoder_outputs, decoder_state):
    # Assuming encoder_outputs is [batch_size, input_steps, features]
    # and decoder_state is [batch_size, features]
    score = Dense(encoder_outputs.shape[2])(decoder_state)  # Project decoder state
    score = tf.expand_dims(score, 1)  # Expand dims to add input_steps axis
    score = score + encoder_outputs  # Add to encoder outputs
    attention_weights = Activation("softmax")(score)  # Compute attention weights
    context_vector = tf.reduce_sum(attention_weights * encoder_outputs, axis=1)
    return context_vector


def create_attention_model(input_steps, output_steps, features):
    # Encoder

    encoder_inputs = Input(shape=(input_steps, features))

    encoder_lstm1 = LSTM(
        200,
        return_sequences=True,
        kernel_regularizer=L2(0.01),
        recurrent_regularizer=L2(0.01),
    )
    encoder_output1 = encoder_lstm1(encoder_inputs)

    encoder_lstm_final = LSTM(100, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder_lstm_final(encoder_output1)

    # Decoder
    decoder_initial_input = RepeatVector(output_steps)(
        state_h
    )  # Prepare decoder inputs

    decoder_lstm = LSTM(100, return_sequences=True)
    decoder_output1 = decoder_lstm(
        decoder_initial_input, initial_state=[state_h, state_c]
    )

    # Manually apply attention mechanism for each timestep
    context_vectors_list1 = []
    for t in range(output_steps):
        # Apply attention mechanism
        context_vector_t1 = attention_mechanism(
            encoder_outputs, decoder_output1[:, t, :]
        )
        context_vectors_list1.append(context_vector_t1)

    # Concatenate the list of context vectors
    context_vectors = tf.stack(context_vectors_list1, axis=1)

    # Concatenate context vectors with decoder outputs
    decoder_combined_context1 = Concatenate(axis=-1)([context_vectors, decoder_output1])

    decoder_lstm2 = LSTM(50, return_sequences=True)
    decoder_output2 = decoder_lstm2(decoder_combined_context1)

    # Manually apply attention mechanism for each timestep
    context_vectors_list2 = []
    for t in range(output_steps):
        # Apply attention mechanism
        context_vector_t2 = attention_mechanism(
            encoder_outputs, decoder_output2[:, t, :]
        )
        context_vectors_list2.append(context_vector_t2)

    # Concatenate the list of context vectors
    context_vectors2 = tf.stack(context_vectors_list2, axis=1)
    decoder_combined_context2 = Concatenate(axis=-1)(
        [context_vectors2, decoder_output2]
    )

    # Output layer for reconstruction
    output = TimeDistributed(Dense(1))(decoder_combined_context2)

    # Create and compile the model
    model = Model(inputs=encoder_inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse")  # Use appropriate loss

    return model


def custom_loss_function(y_true, y_pred, past_steps, future_weight):
    """
    Custom loss function that assigns different weights to the errors in predicting
    past and future values in a sequence.

    Parameters:
    y_true (tensor): The true values.
    y_pred (tensor): The predicted values from the model.
    past_steps (int): The number of steps in the sequence corresponding to past values.
    future_steps (int): The number of steps in the sequence corresponding to future values.
    future_weight (float): The weight to assign to the errors in the future values.

    Returns:
    tensor: The computed weighted loss.
    """
    # Split the true and predicted values into past and future parts
    y_true_past, y_true_future = y_true[:, :past_steps], y_true[:, past_steps:]
    y_pred_past, y_pred_future = y_pred[:, :past_steps], y_pred[:, past_steps:]

    # Calculate mean absolute error for past and future parts
    past_loss = tf.keras.losses.mean_squared_error(y_true_past, y_pred_past)
    future_loss = tf.keras.losses.mean_squared_error(y_true_future, y_pred_future)

    if future_weight == 1:
        # If future_weight is 1, calculate normal MAE across the entire sequence
        total_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    else:
        # Weight the future loss and combine it with the past loss
        weighted_future_loss = future_loss * future_weight
        total_loss = tf.reduce_mean(past_loss + weighted_future_loss)

    return total_loss


def filter_by_features(sequence, features, X_feature_dict):
    sequence = deepcopy(sequence)
    feature_indices = [X_feature_dict[feature] for feature in features]
    return sequence[:, :, feature_indices]


def filter_y_by_features(sequence, features, y_feature_dict):
    sequence = deepcopy(sequence)
    feature_indices = [y_feature_dict[feature] for feature in features]
    print(feature_indices)
    return sequence[:, feature_indices]


def save_decoder_initial_weights(model):
    initial_weights = {}
    for layer in model.layers:
        if "input" in layer.name:
            continue
        print(layer.name)
        initial_weights[layer.name] = deepcopy(layer.get_weights())
    return initial_weights


def train_model(
    model,
    data,
    features,
    target_features,
    X_feature_dict,
    y_feature_dict,
    epochs=100,
    batch_size=32,
    lr=0.001,
    early_stopping_patience=20,
    loss="mae",
):
    X_train, y_train, X_test, y_test = deepcopy(data)
    X_train = filter_by_features(X_train, features, X_feature_dict)
    y_train = filter_y_by_features(y_train, target_features, y_feature_dict)
    X_test = filter_by_features(X_test, features, X_feature_dict)
    y_test = filter_y_by_features(y_test, target_features, y_feature_dict)

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_lr{lr}_batch{batch_size}_first"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=False
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tensorboard_callback,early_stopping],
    )
    return model

import pandas as pd 
import numpy as np
def eval_model(X_test, y_test_old, model, num_days=6):

    predicted_y_old = model.predict(X_test)
    predicted_y_old = np.squeeze(predicted_y_old, axis=-1)

    print(predicted_y_old.shape)

    if len(predicted_y_old[0]) > num_days:
        predicted_y_old = predicted_y_old[:,-num_days:]
        print(predicted_y_old.shape)
        y_test_old = y_test_old[:,-num_days:]
    
    predicted_y = np.cumsum(predicted_y_old, axis=1)
    y_test = np.cumsum(y_test_old, axis=1)
    # predicted_y = predicted_y_old
    # y_test = y_test_old
   

    num_days = predicted_y.shape[1]  # Assuming this is the number of days
    print(num_days)
    results = pd.DataFrame(predicted_y, columns=[f'predicted_{i+1}' for i in range(num_days)])

    for i in range(num_days):
        results[f'real_{i+1}'] = y_test[:, i]

    # Generate output string with accuracies
    output_string = f"Cluster Number:\n"
    for i in range(num_days):
        tolerance = 0.05  # Set your tolerance level

        # Modify the condition for 'same_day'
        results['same_day'] = ((results[f'predicted_{i+1}'] > 0) & (results[f'real_{i+1}'] > 0)) | \
                    ((results[f'predicted_{i+1}'] < 0) & (results[f'real_{i+1}'] < 0)) | \
                    (np.abs(results[f'predicted_{i+1}'] - results[f'real_{i+1}']) < tolerance)
        accuracy = round(results['same_day'].mean() * 100,2)

        output_string += (
            f"Accuracy{i+1}D {accuracy}% "
            f"PredictedRet: {results[f'predicted_{i+1}'].mean()} "
            f"ActRet: {results[f'real_{i+1}'].mean()}\n"
        )
    
    output_string += f"Train set length:  Test set length: {len(y_test)}\n"

    return output_string, results, predicted_y_old


import plotly.graph_objects as go
def visualize_future_distribution(results):
    '''
    Create stacked box and whisker plots for the predicted and real values
    '''

    fig = go.Figure()
    print(results.shape)

    for i in range(len(results.columns) //2):

        fig.add_trace(go.Box(y=results[f'predicted_{i+1}'], name=f'Predicted {i}')) 
        fig.add_trace(go.Box(y=results[f'real_{i+1}'], name=f'Real {i}'))

    fig.update_layout(
        title='Future Performance of Cluster',
        xaxis_title='Steps in future',
        yaxis_title='Cumulative Percent Change'
    ) 

    return fig


def find_closest_centroid(new_centroid, centroids_list):
    '''
    Find the closest centroid to a new centroid among a list of 2D centroids.
    
    Parameters:
    - new_centroid: NumPy array of shape (m, n) representing the new 2D centroid.
    - centroids_list: List of NumPy arrays, each of shape (m, n) representing existing 2D centroids.
    
    Returns:
    - Index of the closest centroid in the `centroids_list`.
    '''
    # Initialize a list to store distances
    distances = []
    
    # Calculate the distance from the new centroid to each centroid in the list
    for centroid in centroids_list:
        # Calculate Frobenius norm as the distance
        distance = np.linalg.norm(centroid - new_centroid, ord='fro')
        distances.append(distance)
    
    # Convert distances to a NumPy array for efficient operations
    distances = np.array(distances)
    
    # Return the index of the closest centroid
    return np.argmin(distances)


def reconstruct_sequence_elements(X_train, y_train, X_test, y_test, X_feature_dict, y_feature_dict):
    '''
    For experimental purposes we need to recreate the original sequence element objects 
    However, when doing this we lose information such as ticker, start and end dates, etc.
    '''

    training_seq_elements = [] 
    test_seq_elements = []

    for i in range(len(X_train)):
        seq_element = sp.SequenceElement(
            X_train[i],
            y_train[i],
            X_feature_dict,
            y_feature_dict,
            True,
            None,
            None,
            None,
        )
        training_seq_elements.append(seq_element)
    
    for i in range(len(X_test)):
        seq_element = sp.SequenceElement(
            X_test[i],
            y_test[i],
            X_feature_dict,
            y_feature_dict,
            False,
            None,
            None,
            None,
        )
        test_seq_elements.append(seq_element)

    return training_seq_elements, test_seq_elements