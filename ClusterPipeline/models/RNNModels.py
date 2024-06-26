from keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU,BatchNormalization,Input,Masking,Flatten,Permute,RepeatVector,Multiply,Activation,Concatenate
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from enum import Enum
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os
from django.db import models
import pandas as pd
import numpy as np
from keras.layers import Layer, Lambda
from keras.layers import Activation, Flatten
from tensorflow.keras.regularizers import L1, L2, L1L2
import keras.backend as K
import plotly.graph_objects as go
import tensorflow as tf
import json
from django.db.models.signals import post_delete
from django.dispatch import receiver
import shutil
import statistics
import io


class ModelTypes(Enum):
    Traditional = 1
    SAE = 2
    AE = 3  # Traditional autoendoder
    AN = 4  # Attention network

    def __str__(self):
        return self.name


class RNNModel(models.Model):
    model_features = models.JSONField(default=list)
    summary_string = models.CharField(max_length=1000, default="None")
    model_dir = models.CharField(max_length=1000, default="None")
    cluster = models.ForeignKey(
        "StockCluster", on_delete=models.CASCADE, related_name="RNNModels"
    )
    model_type_str = models.CharField(max_length=1000, default="None")
    num_encoder_layers = models.IntegerField(default=0)
    model_metrics = models.JSONField(default=dict)
    target_feature_type = models.CharField(max_length=1000,default = 'cumulative')
    target_features = models.JSONField(default=list, blank=True)

    def initialize(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        model_type=None,
        model_features=None,
        model_dir=None,
        num_autoencoder_layers=None,
        num_encoder_layers=None,
    ):
        self.layers = []
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if model_type is not None:
            self.modelType = model_type
            self.model_type_str = str(model_type)
        if model_dir is not None:
            self.model_dir = model_dir

        self.input_shape = X_train.shape
        self.output_shape = y_train.shape[1]

        self.num_layers = 0

        if model_features is not None:
            self.model_features = model_features

        if num_autoencoder_layers is not None:
            self.num_autoencoder_layers = num_autoencoder_layers
        if num_encoder_layers is not None:
            self.num_encoder_layers = num_encoder_layers

    def add_elements(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def addLSTMLayer(self, units, return_sequences=True, activation="tanh"):
        if self.num_layers == 0:
            if self.modelType == ModelTypes.SAE:
                self.layers.append(
                    LSTM(
                        units,
                        input_shape=(None, self.input_shape[2]),
                        return_sequences=return_sequences,
                        activation=activation,
                        name="LSTM_1",
                    )
                )
            else:
                self.layers.append(
                    LSTM(
                        units,
                        input_shape=(None, self.input_shape[2]),
                        return_sequences=return_sequences,
                        activation=activation,
                        name="LSTM_1",
                    )
                )
        else:
            self.layers.append(
                LSTM(
                    units,
                    return_sequences=return_sequences,
                    activation=activation,
                    name="LSTM_" + str(self.num_layers + 1),
                )
            )

        self.num_layers += 1

    def addGRULayer(self, units, return_sequences=True, activation="tanh"):
        if self.num_layers == 0:
            if self.modelType == ModelTypes.SAE:
                self.layers.append(
                    LSTM(
                        units,
                        input_shape=(None, self.input_shape[2]),
                        return_sequences=return_sequences,
                        activation=activation,
                        name="GRU_1",
                    )
                )
            else:
                self.layers.append(
                    LSTM(
                        units,
                        input_shape=(None, self.input_shape[2]),
                        return_sequences=return_sequences,
                        activation=activation,
                        name="GRU_1",
                    )
                )
        else:
            self.layers.append(
                GRU(
                    units,
                    return_sequences=return_sequences,
                    activation=activation,
                    name="GRU_" + str(self.num_layers + 1),
                )
            )

        self.num_layers += 1

    # def addEchoStateLayer(self,units,return_sequences=True,activation='tanh'):
    #     if self.num_layers == 0:
    #         self.model.add(EchoState(units, input_shape=self.input_shape,return_sequences=return_sequences,activation=activation),name = 'EchoState_1')
    #     else:
    #         self.model.add(EchoState(units,return_sequences=return_sequences,activation=activation),name = 'EchoState_'+str(self.num_layers+1))

    #     self.num_layers += 1

    def addDropoutLayer(self, rate):
        self.layers.append(Dropout(rate, name="Dropout_" + str(self.num_layers)))

    def addBatchNormLayer(self):
        self.layers.append(BatchNormalization(name="BatchNorm_" + str(self.num_layers)))

    def buildAutoencoderBlock(self):
        encoder = Sequential()
        encoder.add(
            LSTM(
                self.input_shape[2],
                input_shape=(self.input_shape[1], self.input_shape[2]),
                return_sequences=False,
            )
        )
        encoder.add(RepeatVector(self.input_shape[1]))

        decoder = Sequential()
        decoder.add(LSTM(self.input_shape[2], return_sequences=True))

        return Sequential([encoder, decoder])

    def buildModel(self):
        new_model = Sequential()
        if self.modelType == ModelTypes.Traditional:
            for layer in self.layers:
                new_model.add(layer)
            self.addLSTMLayer(units=50, return_sequences=False)
            new_model.add(Dense(units=6))

        elif self.modelType == ModelTypes.SAE:
            for i in range(self.num_autoencoder_layers):
                auto_encoder = self.buildAutoencoderBlock()
                new_model.add(auto_encoder)

            for layer in self.layers:
                new_model.add(layer)

        elif self.modelType == ModelTypes.AE:
            cur_encoder_layers = 0
            total_layers = 0

            while cur_encoder_layers < self.num_encoder_layers:
                next_layer = self.layers[total_layers]
                if (
                    next_layer.name.split("_")[0] == "LSTM"
                    or next_layer.name.split("_")[0] == "GRU"
                ):
                    cur_encoder_layers += 1
                new_model.add(next_layer)
                total_layers += 1

            for i in range(total_layers, len(self.layers)):
                if (
                    self.layers[i].name.split("_")[0] == "LSTM"
                    or self.layers[i].name.split("_")[0] == "GRU"
                ):
                    break
                new_model.add(self.layers[i])
                total_layers += 1

            new_model.add(RepeatVector(self.output_shape, name="repeat_vector"))

            for i in range(total_layers, len(self.layers)):
                new_model.add(self.layers[i])

            new_model.add(TimeDistributed(Dense(1), name="time_distributed_output"))

        new_model.compile(loss="mse", optimizer=Adam(learning_rate=0.0005))
        self.model = new_model

    def fit(self, epochs=100, batch_size=5):
        # After building the model

        summary_string_list = []

        self.model, self.test_model = create_attention_model(
            input_steps=self.input_shape[1], output_steps=self.output_shape, features=self.input_shape[2]
        )

        # self.model.summary(print_fn=lambda x: summary_string_list.append(x))
        # self.summary_string = "\n".join(summary_string_list).replace('"', '\\"')
        # self.summary_string = self.summary_string.replace("'", "\\'")
        # self.summary_string = self.summary_string.replace("\n", "\\n")
        self.summary_string = " "

        # Before training, inspect the shape of training and validation data
        print("Training data shape:", self.X_train.shape, self.y_train.shape)
        print("Validation data shape:", self.X_test.shape, self.y_test.shape)

        # Optionally, implement a custom training loop for further debugging

        patience = 15
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Metric to monitor (e.g., validation loss)
            patience=patience,  # Number of epochs with no improvement before stopping
            restore_best_weights=True,  # Restore model weights to the best epoch
        )

        if not os.path.exists(self.model_dir + "/log_dir"):
            os.makedirs(self.model_dir + "/log_dir")

        callbacks = [
            TensorBoard(log_dir=self.model_dir + "/log_dir", histogram_freq=1),
        ]
        callbacks.append(early_stopping)

        # Train the model with early stopping

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            shuffle = False
        )

        stopped_epoch = early_stopping.stopped_epoch
        effective_epochs = stopped_epoch - patience + 1

        error = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        self.model_metrics = {
            "effective_epochs": effective_epochs,
            "error": round(error, 2),
        }

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_test(self):
        predicted_y = self.model.predict(self.X_test)
        predicted_y = np.squeeze(predicted_y, axis=-1)

        if self.target_feature_type == 'lag':
            predicted_y = predicted_y[:,-25:]
            self.y_test = self.y_test[:,-25:]

            predicted_y = np.cumsum(predicted_y,axis=1)
            self.y_test = np.cumsum(self.y_test,axis=1)

        num_days = predicted_y.shape[1]  # Assuming this is the number of days
        results = pd.DataFrame(
            predicted_y, columns=[f"predicted_{i+1}" for i in range(num_days)]
        )

        for i in range(num_days):
            results[f"real_{i+1}"] = self.y_test[:, i]

        # Calculate the P/L for each predictio
        # Generate output string with accuracies
        self.step_results = []
        for i in range(num_days):
            step_result = StepResult.objects.create(
                steps_in_future=i + 1,
                RNNModel=self,
                train_set_length=len(self.X_train),
                test_set_length=len(self.y_test),
            )
            same_day = (
                (results[f"predicted_{i+1}"] > 0) & (results[f"real_{i+1}"] > 0)
            ) | ((results[f"predicted_{i+1}"] < 0) & (results[f"real_{i+1}"] < 0))
            accuracy = round(same_day.mean() * 100, 2)
            w_accuracy = round(
                weighted_dir_acc(results[f"predicted_{i+1}"], results[f"real_{i+1}"]), 2
            )
            p_l = profit_loss(results[f"predicted_{i+1}"], results[f"real_{i+1}"])

            step_result.predicted_values = list(results[f"predicted_{i+1}"])
            step_result.actual_values = list(results[f"real_{i+1}"])

            step_result.dir_accuracy = accuracy
            step_result.p_l = p_l
            step_result.weighted_dir_acc = w_accuracy
            step_result.predicted_return = round(results[f"predicted_{i+1}"].mean(), 2)
            step_result.actual_return = round(results[f"real_{i+1}"].mean(), 2)
            step_result.save()

    def serialize(self):
        """
        Method to serialize the cluster. This method saves the model and the sequences to the database
        """

        # Check if the directory exists
        if not os.path.exists(self.model_dir):
            # Create the directory if it doesn't exist
            print("Creating directory " + self.model_dir)
            os.makedirs(self.model_dir)

        # Save the model
        self.test_model.save(self.model_dir + "model.h5")
        del self.model # Delete the model to save memory
        del self.test_model

    def deserialize_model(self):
        """
        Method to load the model from the database
        """
        self.model = tf.keras.models.load_model(self.model_dir + "model.h5")

    def generate_results(self):
        self.step_results = self.model_results.all()
        results = {
            "train_set_length": self.step_results[0].train_set_length,
            "test_set_length": self.step_results[0].test_set_length,
            "cluster_label": int(self.cluster.label),
            "step_accuracy": [],
            "step_accuracy_weighted": [],
            "step_predicted_return": [],
            "step_actual_return": [],
            "step_p_l": [],
        }

        for result in self.step_results:
            results["step_accuracy"].append(int(result.dir_accuracy))
            results["step_accuracy_weighted"].append(int(result.weighted_dir_acc))
            results["step_predicted_return"].append(float(result.predicted_return))
            results["step_actual_return"].append(float(result.actual_return))
            results["step_p_l"].append(float(result.p_l))
        return results

    def generate_model_config(self):
        model_config = {
            "model_features": self.model_features,
            "num_encoder_layers": self.num_encoder_layers,
            "summary_string": self.summary_string,
        }
        return model_config

    def filter_results(self, threshhold=0.2, test_set_length=30):
        self.step_results = StepResult.objects.filter(RNNModel=self)
        for result in self.step_results:
            if (
                result.dir_accuracy < threshhold
                or result.test_set_length < test_set_length
            ):
                result.delete()
        self.num_results = len(self.step_results)

    def compute_average_accuracy(self):
        self.step_results = StepResult.objects.filter(RNNModel=self)
        self.num_results = len(self.step_results)
        self.avg_accuracy = 0
        if self.num_results == 0:
            self.model_metrics["avg_accuracy"] = self.avg_accuracy
            return 0

        for result in self.step_results:
            self.avg_accuracy += result.dir_accuracy
        self.avg_accuracy /= self.num_results

        self.model_metrics["avg_accuracy"] = self.avg_accuracy
        return self.avg_accuracy

    def visualize_future_distribution(self, isTest=True):
        """
        Create stacked box and whisker plots for the predicted and real values
        """

        fig = go.Figure()
        step_results = StepResult.objects.filter(RNNModel=self)

        for step_result in step_results:
            i = step_result.steps_in_future

            if isTest:
                fig.add_trace(
                    go.Box(y=step_result.predicted_values, name=f"Predicted {i}")
                )
                fig.add_trace(go.Box(y=step_result.actual_values, name=f"Real {i}"))
            else:
                fig.add_trace(go.Box(y=self.y_train[:, i - 1], name=f"TrainSet {i}"))
                fig.add_trace(go.Box(y=self.y_test[:, i - 1], name=f"TestSet {i}"))

        fig.update_layout(
            title="Future Performance of Cluster",
            xaxis_title="Steps in future",
            yaxis_title="Cumulative Percent Change",
        )

        return fig

    def create_prediction(
        self, prediction_values, prediction_dates, stock_prediction, prev_day_price
    ):
        """
        Create a prediction object
        """
        print("Creating prediction")
        print(prediction_values)
        print(prediction_dates)
        print(stock_prediction.id)
        print(prev_day_price)

        model_prediction = ModelPrediction.objects.create(
            rnn_model=self,
            stock_prediction=stock_prediction,
            predicted_values=prediction_values,
            prediction_dates=prediction_dates,
            prev_day_price=prev_day_price,
        )
        model_prediction.initialize()
        model_prediction.save()

        return model_prediction


@receiver(post_delete, sender=RNNModel)
def delete_model(sender, instance, **kwargs):
    """
    Delete the model when the RNNModel is deleted
    """
    if os.path.exists(instance.model_dir):
        print("Deleting model " + instance.model_dir)
        try:
            shutil.rmtree(instance.model_dir)
            print(
                f"Directory '{instance.model_dir}' and all its contents have been removed."
            )
        except OSError as error:
            print(f"Error: {error}")


def create_modelAE(input_shape, latent_dim=6):
    # Input layer
    input_layer = Input(shape=(None, input_shape),name = 'input_layer')

    # masking_layer = Masking(mask_value=0.0, name='masking_layer')(input_layer)

    # Encoder
    encoder_lstm1 = LSTM(units=50, activation='tanh', return_state=True,return_sequences=True,
                     name='encoder_lstm_1_freeze', kernel_regularizer=L2(.001), recurrent_regularizer=L2(.001))
    encoder_outputs1 = encoder_lstm1(input_layer)
    encoder_states1 = encoder_outputs1[1:]

    encoder_lstm2 = LSTM(units=25, activation='tanh', return_state=True,return_sequences=True, name = 'encoder_lstm_2_freeze',
                         )
    encoder_outputs2 = encoder_lstm2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]

    encoder_lstm3 = LSTM(units=15, activation='tanh', return_state=True,return_sequences=True, name='encoder_lstm_3_freeze')
    encoder_outputs3 = encoder_lstm3(encoder_outputs2[0])
    encoder_states3 = encoder_outputs3[1:]

    encoder_lstm4 = LSTM(units=10, activation='tanh', return_state=True,return_sequences=True, name='encoder_lstm_4_restore')
    encoder_outputs4 = encoder_lstm4(encoder_outputs3[0])
    encoder_states4 = encoder_outputs4[1:]


    # attention = Dense(1, activation='tanh')(encoder_lstm4)
    # attention = Flatten()(attention)
    # attention_weights = Activation('softmax')(attention)
    # context = Multiply()([encoder_lstm4, Permute([2, 1])(RepeatVector(6)(attention_weights))])

    decoder_inputs = RepeatVector(15, name='repeat_vector')(encoder_states4[0])
    

    # Decoder
    decoder_lstm1 = LSTM(units=50, activation='tanh', return_sequences=True, name='decoder_lstm_1_freeze',
    
                        )(decoder_inputs, initial_state=encoder_states1)
    decoder_lstm2 = LSTM(units=25, activation='tanh', return_sequences=True, name='decoder_lstm_2_freeze',
                        )(decoder_lstm1, initial_state=encoder_states2)
    decoder_lstm3 = LSTM(units=15, activation='tanh', return_sequences=True, name='decoder_lstm_3_freeze',
                         )(decoder_lstm2, initial_state=encoder_states3)
    decoder_lstm4 = LSTM(units=10, activation='tanh', return_sequences=True, name='decoder_lstm_4_restore',
                         )(decoder_lstm3, initial_state=encoder_states4)

    # decoder_lstm3 = LSTM(units=5, activation='tanh', return_sequences=True, name='decoder_lstm_3_restore',
    #                      )(decoder_dropout2)
    # decoder_dropout3 = Dropout(0.2, name='decoder_dropout_3_restore')(decoder_lstm3)

    

    time_distributed_output = TimeDistributed(Dense(1), name='time_distributed_output')(decoder_lstm4)

    # final_output = time_distributed_output[:, -6:, :]

    # Create the model
    model_lstm = Model(inputs=input_layer, outputs=time_distributed_output)

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model_lstm.compile(optimizer=optimizer, loss="mae")

    return model_lstm


def attention_mechanism(encoder_outputs, decoder_state):
    # Assuming encoder_outputs is [batch_size, input_steps, features]
    # and decoder_state is [batch_size, features]
    score = Dense(encoder_outputs.shape[2])(decoder_state)  # Project decoder state
    score = tf.expand_dims(score, 1)  # Expand dims to add input_steps axis
    score = score + encoder_outputs  # Add to encoder outputs
    attention_weights = Activation("softmax")(score)  # Compute attention weights
    context_vector = tf.reduce_sum(attention_weights * encoder_outputs, axis=1)
    return context_vector, attention_weights

def no_training_output(tensor):
    return K.stop_gradient(tensor)  # This halts gradients for the tensor


def create_attention_model(input_steps, output_steps, features):
    # Encoder

    encoder_inputs = Input(shape=(input_steps, features), name='input')

    encoder_lstm1 = LSTM(
        200,
        return_sequences=True,
        kernel_regularizer=L2(0.00),
        recurrent_regularizer=L2(0.001),
        name="encoder_lstm_1_freeze",
    )
    encoder_output1 = encoder_lstm1(encoder_inputs)

    encoder_lstm_final = LSTM(100, return_state=True, return_sequences=True, name="encoder_lstm_final_freeze")
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
        context_vector_t1, attention_weights_t1 = attention_mechanism(
            encoder_outputs, decoder_output1[:, t, :]
        )
        context_vectors_list1.append(context_vector_t1)

    # Concatenate the list of context vectors
    context_vectors = tf.stack(context_vectors_list1, axis=1)

    # Concatenate context vectors with decoder outputs
    decoder_combined_context1 = Concatenate(axis=-1)([context_vectors, decoder_output1])

    decoder_lstm2 = LSTM(200, return_sequences=True)
    decoder_output2 = decoder_lstm2(decoder_combined_context1)

    # Manually apply attention mechanism for each timestep
    context_vectors_list2 = []
    attention_weights_list2 = []
    for t in range(output_steps):
        # Apply attention mechanism
        context_vector_t2, attention_weights_t2 = attention_mechanism(
            encoder_outputs, decoder_output2[:, t, :]
        )
        context_vectors_list2.append(context_vector_t2)
        attention_weights_list2.append(attention_weights_t2)


    # Concatenate the list of context vectors
    context_vectors2 = tf.stack(context_vectors_list2, axis=1)
    decoder_combined_context2 = Concatenate(axis=-1)(
        [context_vectors2, decoder_output2]
    )
    attention_weights = tf.stack(attention_weights_list2, axis=1)

    attention_weights_output = Lambda(lambda x: K.stop_gradient(x))(attention_weights)

    # Output layer for reconstruction
    # output = TimeDistributed(Dense(1))(decoder_combined_context2)

    main_output = TimeDistributed(Dense(1))(decoder_combined_context2)

    # Create and compile the model
    training_model = Model(inputs=encoder_inputs, outputs=main_output)
    training_model.compile(optimizer="adam", loss="mse")  # Use appropriate loss

    test_model = Model(inputs=encoder_inputs, outputs=[main_output, attention_weights_output])
    test_model.compile(optimizer="adam", loss="mse")  # Use appropriate loss

    return training_model, test_model

class StepResult(models.Model):
    steps_in_future = models.IntegerField(default=0)
    RNNModel = models.ForeignKey(
        RNNModel, on_delete=models.CASCADE, related_name="model_results"
    )
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
    return p_l / len(predicted)


def weighted_dir_acc(predicted, actual):
    directional_accuracy = (np.sign(predicted) == np.sign(actual)).astype(int)
    magnitude_difference = np.abs(np.abs(predicted) - np.abs(actual)) + 1e-6
    weights = np.abs(actual) / magnitude_difference
    return np.sum(directional_accuracy * weights) / np.sum(weights) * 100


class ModelPrediction(models.Model):
    rnn_model = models.ForeignKey(
        RNNModel, on_delete=models.CASCADE, related_name="rnn_model_predictions"
    )
    stock_prediction = models.ForeignKey(
        "StockPrediction",
        on_delete=models.CASCADE,
        related_name="stock_model_predictions",
    )
    predicted_values = models.JSONField(default=list, blank=True)
    prediction_dates = models.JSONField(default=list, blank=True)
    actual_values = models.JSONField(default=list, blank=True)
    status = models.BooleanField(default=True)
    cluster_id = models.IntegerField(default=None, blank=True, null=True)
    group_id = models.IntegerField(default=None, blank=True, null=True)
    filtered_accuracy = models.FloatField(default=0)
    prev_day_price = models.FloatField(default=0)
    effective_epochs = models.IntegerField(default=0)
    start_date = models.DateTimeField(default=None, blank=True, null=True)
    end_date = models.DateTimeField(default=None, blank=True, null=True)

    def initialize(self):
        self.cluster_id = self.rnn_model.cluster.pk
        self.group_id = self.rnn_model.cluster.cluster_group.pk
        step_results = list(StepResult.objects.filter(RNNModel=self.rnn_model))
        filtered_predicted_values = [
            step.dir_accuracy
            for step, pred in zip(step_results, self.predicted_values)
            if pred is not None
        ]
        self.filtered_accuracy = statistics.mean(filtered_predicted_values)
        self.effective_epochs = self.rnn_model.model_metrics["effective_epochs"]
        self.start_date = self.prediction_dates[0]
        self.end_date = self.prediction_dates[-1]

    def create_model_pred_dict(self):
        model_dict = {
            "model_prediction_id": self.pk,
            "model_id": self.rnn_model.pk,
            "stock_prediction_id": self.stock_prediction.pk,
            "group_id": self.group_id,
            "cluster_id": self.cluster_id,
            "filtered_accuracy": self.filtered_accuracy,
            "effective_epochs": self.effective_epochs,
            "start_date": self.start_date.strftime("%Y-%m-%d: %H:%M:%S"),
            "end_date": self.end_date.strftime("%Y-%m-%d: %H:%M:%S"),
            "predicted_values": self.predicted_values,
            "predicted_dates": self.prediction_dates,
            "prev_day_price": self.prev_day_price,
            "status": self.status,
        }
        return model_dict

    def update_prediction(self, model_pred_dict):
        self.status = model_pred_dict["status"]
