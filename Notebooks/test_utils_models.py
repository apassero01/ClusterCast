from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from tensorflow.keras.layers import GRU, LSTM, RNN, Dense, Dropout, GRUCell, LSTMCell
from tensorflow.keras import Model, Input
from tensorflow.keras.regularizers import L2

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LayerNormalization

# @misc{tfts2020,
#   author = {Longxing Tan},
#   title = {Time series prediction},
#   year = {2020},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/longxingtan/time-series-prediction}},
# }

class FullAttention(tf.keras.layers.Layer):
    """Multi-head attention layer"""

    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float = 0.0) -> None:
        """Initialize the layer.

        Parameters:
        -----------
        hidden_size : int
            The number of hidden units in each attention head.
        num_heads : int
            The number of attention heads.
        attention_dropout : float, optional
            Dropout rate for the attention weights. Defaults to 0.0.
        """
        super(FullAttention, self).__init__()
        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({}).".format(hidden_size, num_heads)
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.dense_q = Dense(self.hidden_size, use_bias=False)
        self.dense_k = Dense(self.hidden_size, use_bias=False)
        self.dense_v = Dense(self.hidden_size, use_bias=False)
        self.dropout = Dropout(rate=self.attention_dropout)
        super(FullAttention, self).build(input_shape)

    def call(self, q, k, v, mask=None):
        """use query and key generating an attention multiplier for value, multi_heads to repeat it

        Parameters
        ----------
        q : tf.Tenor
            Query with shape batch * seq_q * fea
        k : tf.Tensor
            Key with shape batch * seq_k * fea
        v : tf.Tensor
            value with shape batch * seq_v * fea
        mask : _type_, optional
            important to avoid the leaks, defaults to None, by default None

        Returns
        -------
        tf.Tensor
            tensor with shape batch * seq_q * (units * num_heads)
        """
        q = self.dense_q(q)  # project the query/key/value to num_heads * units
        k = self.dense_k(k)
        v = self.dense_v(v)

        q_ = tf.concat(tf.split(q, self.num_heads, axis=2), axis=0)  # multi-heads transfer to multi-sample
        k_ = tf.concat(tf.split(k, self.num_heads, axis=2), axis=0)
        v_ = tf.concat(tf.split(v, self.num_heads, axis=2), axis=0)

        score = tf.linalg.matmul(q_, k_, transpose_b=True)  # => (batch * heads) * seq_q * seq_k
        score /= tf.cast(tf.shape(q_)[-1], tf.float32) ** 0.5

        if mask is not None:
            score += (mask * -1e9)

        weights = tf.nn.softmax(score)
        weights = self.dropout(weights)

        weights_reshaped = tf.reshape(weights, [self.num_heads, -1, tf.shape(q)[1], tf.shape(k)[1]])
        # Take the mean across the heads
        weights_mean = tf.reduce_mean(weights_reshaped, axis=0)

        outputs = tf.linalg.matmul(weights, v_)  
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)

        return outputs, weights_mean  # Now also returning the attention weights

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
        }
        base_config = super(FullAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_type, rnn_size, name, rnn_dropout=0, dense_size=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.rnn_type = rnn_type
        if rnn_type.lower() == "gru":
            self.rnn = GRU(
                name=name,
                units=rnn_size, activation="tanh", return_state=True, return_sequences=True, dropout=rnn_dropout
            )
            self.dense_state = Dense(dense_size, activation="tanh")  # For projecting GRU state
        elif rnn_type.lower() == "lstm":
            self.rnn = LSTM(
                name=name,
                units=rnn_size,
                activation="tanh",
                return_state=True,
                return_sequences=True,
                dropout=rnn_dropout,
                recurrent_regularizer=L2(0.01),
            )
            self.dense_state_h = Dense(dense_size, activation="tanh")  # For projecting LSTM hidden state
            self.dense_state_c = Dense(dense_size, activation="tanh")  # For projecting LSTM cell state
        self.dense = Dense(units=dense_size, activation="tanh")

    def call(self, inputs):
        """Seq2seq encoder

        Parameters
        ----------
        inputs : tf.Tensor
            _description_

        Returns
        -------
        tf.Tensor
            batch_size * input_seq_length * rnn_size, state: batch_size * rnn_size
        """
        if self.rnn_type.lower() == "gru":
            outputs, state = self.rnn(inputs)
            # state = self.dense_state(state)
        elif self.rnn_type.lower() == "lstm":
            outputs, state1, state2 = self.rnn(inputs)
            # state1_projected = self.dense_state_h(state1)  # Project hidden state
            # state2_projected = self.dense_state_c(state2)  # Project cell state
            state = (state1, state2)
        else:
            raise ValueError("No supported rnn type of {}".format(self.rnn_type))
        # encoder_hidden_state = tuple(self.dense(hidden_state) for _ in range(params['num_stacked_layers']))
        outputs = self.dense(outputs)  # => batch_size * input_seq_length * dense_size
        return outputs, state

    def build(self, input_shape):
        super(Encoder, self).build(input_shape)


class Decoder1(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_type="gru",
        rnn_size=32,
        predict_sequence_length=3,
        use_attention=False,
        attention_sizes=32,
        attention_heads=1,
        attention_dropout=0.0,
        **kwargs
    ):
        super(Decoder1, self).__init__(**kwargs)
        self.predict_sequence_length = predict_sequence_length
        self.use_attention = use_attention
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size

        # Initialize RNN Cell based on rnn_type
        if self.rnn_type.lower() == "gru":
            self.rnn_cell = GRUCell(self.rnn_size)
        elif self.rnn_type.lower() == "lstm":
            self.rnn_cell = LSTMCell(units=self.rnn_size)

        # Initialize Dense layer and Attention mechanism if used
        self.dense = Dense(units=1, activation=None)
        if self.use_attention:
            self.attention = FullAttention(
                hidden_size=attention_sizes,
                num_heads=attention_heads,
                attention_dropout=attention_dropout,
            )

    def call(
        self,
        decoder_features,
        decoder_init_input,
        init_state,
        encoder_output,
        teacher=None,
        scheduler_sampling=0,
        training=None
    ):
        decoder_outputs = []
        attention_weights = []  # List to store attention weights
        prev_output = decoder_init_input
        prev_state = init_state

        if teacher is not None:
            teacher = tf.squeeze(teacher, 2)
            teachers = tf.split(teacher, self.predict_sequence_length, axis=1)

        for i in range(self.predict_sequence_length):
            if training:
                p = np.random.uniform(low=0, high=1, size=1)[0]
                this_input = teachers[i] if teacher is not None and p > scheduler_sampling else prev_output
            else:
                this_input = prev_output

            if decoder_features is not None:
                this_input = tf.concat([this_input, decoder_features[:, i]], axis=-1)

            if self.use_attention:
                # Use the hidden state for attention query
                query = tf.expand_dims(prev_state[0] if self.rnn_type.lower() == "lstm" else prev_state, axis=1)
                att_output, weights = self.attention(query, encoder_output, encoder_output)
                att_output = tf.squeeze(att_output, axis=1)
                this_input = tf.concat([this_input, att_output], axis=-1)
                attention_weights.append(weights)

            # Update state based on RNN type
            if self.rnn_type.lower() == "lstm":
                this_output, this_state = self.rnn_cell(this_input, states=prev_state)
            else:  # GRU
                this_output, this_state = self.rnn_cell(this_input, states=[prev_state])

            prev_state = this_state
            prev_output = self.dense(this_output)
            decoder_outputs.append(prev_output)

        decoder_outputs = tf.concat(decoder_outputs, axis=1)
        decoder_outputs = tf.expand_dims(decoder_outputs, -1)
        attention_weights = tf.concat(attention_weights, axis=1)

        print("decoder outputs")
        print(decoder_outputs.shape)

        return decoder_outputs, attention_weights
    
class DecoderProb(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_type="gru",
        rnn_size=32,
        predict_sequence_length=3,
        use_attention=False,
        attention_sizes=32,
        attention_heads=1,
        attention_dropout=0.0,
        **kwargs
    ):
        super(DecoderProb, self).__init__(**kwargs)
        self.predict_sequence_length = predict_sequence_length
        self.use_attention = use_attention
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size

        # Initialize RNN Cell based on rnn_type
        if self.rnn_type.lower() == "gru":
            self.rnn_cell = GRUCell(self.rnn_size)
        elif self.rnn_type.lower() == "lstm":
            self.rnn_cell = LSTMCell(units=self.rnn_size)

        # Initialize Dense layer and Attention mechanism if used
        self.dense = Dense(units=2, activation=None)  # Updated to output 2 values (mean and variance)
        if self.use_attention:
            self.attention = FullAttention(
                hidden_size=attention_sizes,
                num_heads=attention_heads,
                attention_dropout=attention_dropout,
            )

    def call(
        self,
        decoder_features,
        decoder_init_input,
        init_state,
        encoder_output,
        teacher=None,
        scheduler_sampling=0,
        training=None
    ):
        print("ENCODER OUTPUT SHAPE")
        print(encoder_output.shape)
        decoder_outputs = []
        attention_weights = []  # List to store attention weights
        prev_output = decoder_init_input
        prev_state = init_state

        if teacher is not None:
            teacher = tf.squeeze(teacher, 2)
            teachers = tf.split(teacher, self.predict_sequence_length, axis=1)

        for i in range(self.predict_sequence_length):
            if training:
                p = np.random.uniform(low=0, high=1, size=1)[0]
                this_input = teachers[i] if teacher is not None and p > scheduler_sampling else prev_output
            else:
                this_input = prev_output
            

            if decoder_features is not None:
                this_input = tf.concat([this_input, decoder_features[:, i]], axis=-1)

            if self.use_attention:
                # Use the hidden state for attention query
                query = tf.expand_dims(prev_state[0] if self.rnn_type.lower() == "lstm" else prev_state, axis=1)
                att_output, weights = self.attention(query, encoder_output, encoder_output)
                att_output = tf.squeeze(att_output, axis=1)
                this_input = tf.concat([this_input, att_output], axis=-1)
                attention_weights.append(weights)

            # Update state based on RNN type
            if self.rnn_type.lower() == "lstm":
                this_output, this_state = self.rnn_cell(this_input, states=prev_state)
            else:  # GRU
                this_output, this_state = self.rnn_cell(this_input, states=[prev_state])

            prev_state = this_state
            prev_output = self.dense(this_output)  # Output is now of shape (None, 2)
            decoder_outputs.append(prev_output)

        print("BEFORE")
        print(decoder_outputs[0].shape)
        decoder_outputs = tf.stack(decoder_outputs, axis=1)
        attention_weights = tf.concat(attention_weights, axis=1)

        return decoder_outputs, attention_weights


def create_seq2seq(predict_sequence_length, input_steps, input_features):
    encoder_input = Input(shape=(input_steps, input_features))
    
    # Initialize the encoder
    encoder = Encoder(rnn_type="lstm", rnn_size=64, rnn_dropout=0.2, dense_size=64)
    encoder_output, encoder_state = encoder(encoder_input)
    
    # Prepare decoder initial input (e.g., a tensor of zeros)
    # This can be modified based on the specific requirements of your task
    decoder_init_input = tf.zeros_like(encoder_input[:, 0, :1])  # Assuming the decoder starts with zeros
    
    # Initialize the decoder
    # Adjust these parameters based on your specific requirements
    decoder = Decoder1(
        rnn_type="lstm",
        rnn_size=32,
        predict_sequence_length=predict_sequence_length,
        use_attention=True,
        attention_sizes=32,
        attention_heads=1,
        attention_dropout=0
    )
    
    # Call the decoder
    # You might need to adjust the parameters passed to the decoder based on your specific implementation and requirements
    decoder_output = decoder(
        decoder_features=None,  # If you have additional features for the decoder, provide them here
        decoder_init_input=decoder_init_input,
        init_state=encoder_state,  # Pass the encoder state as the initial state for the decoder
        teacher=None,  # If you're using teacher forcing during training, provide the target sequences here
        scheduler_sampling=0,  # Adjust this for scheduled sampling (if used)
        encoder_output=encoder_output,  # Pass the encoder output for attention
    )
    
    # Create the model
    model = Model(inputs=encoder_input, outputs=decoder_output)
    
    # Compile the model
    # You can change the optimizer and loss function based on your specific requirements
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


class CustomSeq2SeqModel(tf.keras.Model):
    def __init__(self, predict_sequence_length, input_steps, input_features, **kwargs):
        super(CustomSeq2SeqModel, self).__init__(**kwargs)
        self.encoder1 = Encoder(rnn_type="lstm", name = 'encoder1', rnn_size=64, rnn_dropout=0.2, dense_size=63)
        # self.encoder2 = Encoder(rnn_type="lstm", name = 'encoder2', rnn_size=128, rnn_dropout=0.2, dense_size=128)
        self.decoder1 = Decoder1(
            rnn_type="lstm",
            rnn_size=64,
            predict_sequence_length=predict_sequence_length,
            use_attention=True,
            attention_sizes=16,
            attention_heads=1,
            attention_dropout=.1
        )
        # self.decoder2 = Decoder1(
        #     rnn_type="lstm",
        #     rnn_size=256,
        #     predict_sequence_length=predict_sequence_length,
        #     use_attention=True,
        #     attention_sizes=64,
        #     attention_heads=1,
        #     attention_dropout=0
        # )

    def call(self, inputs, training=False):
        encoder_output1, encoder_state1 = self.encoder1(inputs)
        # encoder_output2, encoder_state2  = self.encoder2(encoder_output1)

        print(encoder_output1.shape)
        
        # decoder_input = tf.zeros_like(inputs[:, 0, :1])  # Example for initialization
        decoder_input = inputs[:, -1, 0:1]
        decoder_output1, attention_weights_1 = self.decoder1(
            decoder_features=None,
            decoder_init_input=decoder_input,
            init_state=encoder_state1,
            encoder_output=encoder_output1,
            teacher=None,
            scheduler_sampling=0,
            training=training
        )
        # decoder_output2, attention_weights_2 = self.decoder2(
        #     decoder_features=decoder_output1,  # Pass the output from decoder1
        #     decoder_init_input=decoder_input,
        #     init_state=encoder_state1,
        #     encoder_output=decoder_output1,
        #     teacher=None,
        #     scheduler_sampling=0,
        #     training=training
        # )

        
        if training:
            # During training, return only the decoder output.
            return decoder_output1
        else:
            # During evaluation or inference, return both output and attention weights.
            print("Attention Weights Shape: ", attention_weights_1.shape)
            return decoder_output1, attention_weights_1
    
    def freeze_encoder(self):
        self.encoder1.trainable = False
        # self.encoder2.trainable = False


class CustomSeq2SeqModelProb(tf.keras.Model):
    def __init__(self, predict_sequence_length, input_steps, input_features, **kwargs):
        super(CustomSeq2SeqModelProb, self).__init__(**kwargs)
        self.encoder1 = Encoder(rnn_type="lstm", rnn_size=256, rnn_dropout=0.3, dense_size=64)
        # self.encoder2 = Encoder(rnn_type="lstm", rnn_size=128, rnn_dropout=0.3, dense_size=64)
        self.decoder1 = DecoderProb(
            rnn_type="lstm",
            rnn_size=256,
            predict_sequence_length=predict_sequence_length,
            use_attention=True,
            attention_sizes=32,
            attention_heads=2,
            attention_dropout=.2
        )
        # self.decoder2 = DecoderProb(
        #     rnn_type="lstm",
        #     rnn_size=256,
        #     predict_sequence_length=predict_sequence_length,
        #     use_attention=True,
        #     attention_sizes=32,
        #     attention_heads=2,
        #     attention_dropout=.2
        # )

    def call(self, inputs, training=False):
        encoder_output1, encoder_state1 = self.encoder1(inputs)
        # encoder_output2, encoder_state2  = self.encoder2(encoder_output1)

        print(encoder_output1.shape)
        
        decoder_input = tf.zeros_like(inputs[:, 0, :2])  # Example for initialization
        decoder_output1, attention_weights_1 = self.decoder1(
            decoder_features=None,
            decoder_init_input=decoder_input,
            init_state=encoder_state1,
            encoder_output=encoder_output1,
            teacher=None,
            scheduler_sampling=0,
            training=training
        )
        # decoder_output2, attention_weights_2 = self.decoder2(
        #     decoder_features=decoder_output1,  # Pass the output from decoder1
        #     decoder_init_input=decoder_input,
        #     init_state=encoder_state1,
        #     encoder_output=decoder_output1,
        #     teacher=None,
        #     scheduler_sampling=0,
        #     training=training
        # )

        variance_softplus = softplus(decoder_output1[:, :, 1])
        decoder_output1 = tf.concat([decoder_output1[:, :, 0:1], variance_softplus[:, :, tf.newaxis]], axis=-1)
        print("decoder_output")
        print(decoder_output1.shape)
        
        if training:
            # During training, return only the decoder output.
            return decoder_output1
        else:
            # During evaluation or inference, return both output and attention weights.
            print("Attention Weights Shape: ", attention_weights_1.shape)
            return decoder_output1, attention_weights_1
        

class CustomSeq2SeqModelStop(tf.keras.Model):
    def __init__(self, predict_sequence_length, input_steps, input_features, **kwargs):
        super(CustomSeq2SeqModelStop, self).__init__(**kwargs)
        self.encoder1 = Encoder(rnn_type="lstm", name = 'encoder1', rnn_size=64, rnn_dropout=0.2, dense_size=32)
        # self.encoder2 = Encoder(rnn_type="lstm", name = 'encoder2', rnn_size=64, rnn_dropout=0.2, dense_size=128)
        self.decoder1 = Decoder1(
            rnn_type="lstm",
            rnn_size=64,
            predict_sequence_length=predict_sequence_length,
            use_attention=True,
            attention_sizes=32,
            attention_heads=1,
            attention_dropout=0
        )
        # self.decoder2 = Decoder1(
        #     rnn_type="lstm",
        #     rnn_size=128,
        #     predict_sequence_length=predict_sequence_length,
        #     use_attention=True,
        #     attention_sizes=16,
        #     attention_heads=1,
        #     attention_dropout=0
        # )


        self.decoder_stop = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        encoder_output1, encoder_state1 = self.encoder1(inputs)
        # encoder_output2, encoder_state2  = self.encoder2(encoder_output1)

        print(encoder_output1.shape)
        
        # decoder_input = tf.zeros_like(inputs[:, 0, :1])  # Example for initialization
        decoder_input = inputs[:, -1, 0:1]
        decoder_output1, attention_weights_1 = self.decoder1(
            decoder_features=None,
            decoder_init_input=decoder_input,
            init_state=encoder_state1,
            encoder_output=encoder_output1,
            teacher=None,
            scheduler_sampling=0,
            training=training
        )

        # decoder_output2, attention_weights_2 = self.decoder2(
        #     decoder_features=decoder_output1,  # Pass the output from decoder1
        #     decoder_init_input=decoder_input,
        #     init_state=encoder_state1,
        #     encoder_output=encoder_output1,
        #     teacher=None,
        #     scheduler_sampling=0,
        #     training=training
        # )

        ## If you add second layer again you need to figure out how to get the input to the decoder to be correct

        context_vectors = tf.matmul(attention_weights_1, encoder_output1)

        final_input = tf.concat([decoder_output1, context_vectors], axis=-1)

        # Use concatenated output for final decision
        end_decision = self.decoder_stop(final_input)

        # end_decision = self.decoder_stop(decoder_output1)

        
        if training:
            # During training, return only the decoder output.
            return tf.concat([decoder_output1, end_decision], axis=-1)
        else:
            # During evaluation or inference, return both output and attention weights.
            print("Attention Weights Shape: ", attention_weights_1.shape)
            return tf.concat([decoder_output1, end_decision], axis=-1), attention_weights_1
    
    def freeze_encoder(self):
        self.encoder1.trainable = False
        # self.encoder2.trainable = False


def softplus(x):
    return tf.math.log(1 + tf.math.exp(x))

# def gaussian_log_likelihood(labels, predictions):
#     """Computes the Gaussian log likelihood as a loss function.
    
#     Args:
#         labels: A `Tensor` of target values with shape [?, outputsteps, 2],
#                 where labels[:, :, 0] is the true mean and labels[:, :, 1] is the true variance.
#         predictions: A `Tensor` of predicted values with the same shape,
#                      predictions[:, :, 0] is the predicted mean,
#                      predictions[:, :, 1] is the predicted log variance component.
                     
#     Returns:
#         A `Tensor` of the negative Gaussian log likelihood.
#     """
#     true_mean = labels[:, :, 0]
#     true_variance = labels[:, :, 1]

#     pred_mean = predictions[:, :, 0]
#     pred_log_variance = predictions[:, :, 1]
#     pred_variance = 1e-6 + tf.nn.softplus(pred_log_variance)  # ensure non-negative variance
    
#     # Compute the Gaussian log likelihood
#     log_likelihood = -0.5 * (((true_mean - pred_mean) ** 2 / pred_variance) + tf.math.log(pred_variance)) - 0.5 * tf.math.log(2 * np.pi)
    
#     # Return the mean negative log likelihood as the loss
#     return -tf.reduce_mean(log_likelihood)

def gaussian_log_likelihood_with_true_variance(labels, predictions):
    """Computes the Gaussian log likelihood with true variance as a loss function.
    
    Args:
        labels: A `Tensor` of target values with shape [?, outputsteps, 2],
                where labels[:, :, 0] is the true mean and labels[:, :, 1] is the true variance.
        predictions: A `Tensor` of predicted values with the same shape,
                     predictions[:, :, 0] is the predicted mean,
                     predictions[:, :, 1] is the predicted log variance component.
                     
    Returns:
        A `Tensor` of the negative Gaussian log likelihood.
    """
    epsilon = 1e-6  # Small constant for numerical stability
    true_mean = labels[:, :, 0]
    true_log_variance = tf.math.log(labels[:, :, 1] + epsilon)  # Avoid log(0) by adding epsilon

    pred_mean = predictions[:, :, 0]
    pred_log_variance = predictions[:, :, 1]
    pred_variance = tf.exp(pred_log_variance)  # Transform log variance to actual variance

    # Compute the Gaussian log likelihood for the mean
    mean_log_likelihood = -0.5 * (((true_mean - pred_mean) ** 2 / pred_variance) + tf.math.log(pred_variance))

    # Compute the difference in predicted and true log variances
    variance_log_likelihood = -0.5 * (pred_log_variance - true_log_variance) ** 2

    # Combine the likelihoods for mean and variance
    total_loss = mean_log_likelihood + variance_log_likelihood

    # Return the mean negative log likelihood as the loss
    return -tf.reduce_mean(total_loss)
# def gaussian_log_likelihood(labels, predictions):
#     """Computes the Gaussian log likelihood as a loss function.
    
#     Args:
#         labels: A `Tensor` of target values with shape [?, outputsteps, 2],
#                 where labels[:, :, 0] is the true mean and labels[:, :, 1] is the true variance.
#         predictions: A `Tensor` of predicted values with the same shape,
#                      predictions[:, :, 0] is the predicted mean,
#                      predictions[:, :, 1] is the predicted log variance component.
                     
#     Returns:
#         A `Tensor` of the negative Gaussian log likelihood.
#     """
#     true_mean = labels[:, :, 0]
#     true_variance = labels[:, :, 1]

#     pred_mean = predictions[:, :, 0]
#     pred_log_variance = predictions[:, :, 1]
#     pred_variance = 1e-6 + tf.nn.softplus(pred_log_variance)  # ensure non-negative variance
    
#     # Compute the Gaussian log likelihood
#     log_likelihood = -0.5 * (((true_mean - pred_mean) ** 2 / (true_variance + pred_variance)) + tf.math.log(true_variance + pred_variance)) - 0.5 * tf.math.log(2 * np.pi)
    
#     # Return the mean negative log likelihood as the loss
#     return -tf.reduce_mean(log_likelihood)


def cuma_loss_function(y_true, y_pred):
    y_true_cum = tf.cumsum(y_true, axis=1)
    y_pred_cum = tf.cumsum(y_pred, axis=1)

    return tf.reduce_mean(tf.square(y_true_cum - y_pred_cum))

def cumulative_trajectory_loss(y_true, y_pred):
    # Calculate cumulative sums along the time axis (assuming the time axis is axis=1)
    cumsum_true = tf.cumsum(y_true, axis=1)
    cumsum_pred = tf.cumsum(y_pred, axis=1)

    # Calculate the absolute differences
    differences = tf.abs(cumsum_true - cumsum_pred)

    # Sum over the time dimension to approximate the integral of differences
    loss = tf.reduce_mean(tf.reduce_sum(differences, axis=1))
    return loss

from fastdtw import fastdtw

def dtw_distance(x, y):
    # This function computes the DTW distance between two 1-D sequences in NumPy.
    distance, path = fastdtw(x, y)
    return np.float32(distance)


def dtw_loss(y_true, y_pred):
    """
    Compute Dynamic Time Warping Loss between true labels and predictions.
    Each is assumed to be a batch of sequences.

    Parameters:
    y_true (tf.Tensor): True labels.
    y_pred (tf.Tensor): Predictions.

    Returns:
    tf.Tensor: DTW loss for the batch.
    """
    # tf.numpy_function wraps the numpy function to be compatible with TensorFlow.
    # We map dtw_distance across elements of y_true and y_pred.
    # Assume y_true and y_pred are of shape [batch_size, sequence_length]
    return tf.map_fn(
        lambda xy: tf.numpy_function(dtw_distance, [xy[0], xy[1]], Tout=tf.float32),
        (y_true, y_pred),
        dtype=tf.float32
    )


def custom_loss_function(y_true, y_pred, beta=5, financial_penalty_factor=0.1):
    decoder_output = y_pred[:, :, :-1]
    end_decision = y_pred[:, :, -1]  

    # expand end decision shape 
    end_decision = tf.expand_dims(end_decision, axis=2)

    print("End Decision shape ", end_decision.shape)
    print("Decoder Output shape ", decoder_output.shape)


    mse_loss = tf.keras.losses.MeanSquaredError(reduction='none')(y_true, decoder_output)
    mse_loss = tf.expand_dims(mse_loss, axis=-1)  # Add singleton dimension         
    # mse_loss = tf.reduce_mean(mse_loss)
    print("MSE LOSS SHAPE", mse_loss.shape)

    sequence_length = tf.shape(decoder_output)[1]
    # cumulative_stop_decision = tf.cumsum(end_decision, axis=1)
    weights = tf.nn.relu(1.0 - beta * end_decision)
    print("Weights shape ", weights.shape)

    
    weighted_errors = tf.multiply(mse_loss, weights)

    # Reward for longer predictions
    reverse_cumulative_profits = tf.cumsum(tf.abs(y_true[:, ::-1]), axis=1)[:, ::-1]
    print("Reverse Cumulative Profits shape ", reverse_cumulative_profits.shape)
    weights = tf.exp(tf.cast(reverse_cumulative_profits, tf.double) / 10)
    print("financial Weights shape ", weights.shape)
    financial_penalty = financial_penalty_factor * tf.reduce_sum(weights * tf.expand_dims(tf.cast(end_decision, tf.float64), axis=1), axis=1)
    total_loss = tf.reduce_mean(weighted_errors) + tf.reduce_mean(tf.cast(financial_penalty, tf.float32))
    return total_loss



def custom_loss_function_cuma(y_true, y_pred, beta=1, financial_penalty_factor=1):
    decoder_output = y_pred[:, :, :-1]
    end_decision = y_pred[:, :, -1]  

    y_true_cum = tf.cumsum(y_true, axis=1)
    decoder_output = tf.cumsum(decoder_output, axis=1)

    mse_loss = tf.keras.losses.mean_squared_error(y_true_cum, decoder_output)
    print("MSE LOSS", mse_loss.shape)
    # mse_loss = tf.reduce_mean(mse_loss)
    # print("MSE LOSS_REDUCED", mse_loss)

    sequence_length = tf.shape(decoder_output)[1]
    # cumulative_stop_decision = tf.cumsum(end_decision, axis=1)
    print("CUMULATIVE_STOP_DECISION", end_decision.shape)
    weights = tf.nn.relu(1.0 - beta * end_decision)

    print("WEIGHTS", weights)
    
    weighted_errors = mse_loss * weights
    print("WEIGHTED ERROR", weighted_errors.shape)

    # Reward for longer predictions
    reverse_cumulative_profits = tf.cumsum(tf.abs(y_true[:, ::-1]), axis=1)[:, ::-1]
    print("Reverse Cumulative Profits shape ", reverse_cumulative_profits)
    weights_financial = tf.exp(tf.cast(reverse_cumulative_profits, tf.float64) / 10 + 1e-10)
    print("financial Weights shape ", weights_financial.shape)

    financial_penalty = tf.reduce_sum(weights_financial * tf.expand_dims(1 - tf.cast(end_decision, tf.float64), axis=1), axis=1)
    print("Financial Penalty shape ", financial_penalty.shape)
    # Normalization
    mse_norm = weighted_errors / tf.reduce_mean(weighted_errors)
    financial_penalty_norm = financial_penalty / tf.reduce_mean(financial_penalty)
    financial_penalty_norm = financial_penalty_factor * financial_penalty_norm
    print("MSE NORM shape ", mse_norm.shape)
    print("Financial Penalty NORM shape ", financial_penalty_norm.shape)
    

    # Total loss calculation
    total_loss = tf.reduce_mean(tf.cast(mse_norm, tf.float32)) + tf.reduce_mean(tf.cast(financial_penalty_norm, tf.float32))

    return total_loss



def custom_loss(y_true, y_pred_full, financial_penalty = .5):
    # Split predictions and decision signals
    y_pred = y_pred_full[:, :, :-1]  # Predicted day-over-day percent changes
    decision = y_pred_full[:, :, -1:]  # Decision output (already sigmoid)
    print("Decision", decision) 

    # Calculate cumulative sums
    # In real trading, we'd consider each day's change as an independent event where the model decides to 'buy' (predict positive) or 'sell' (predict negative).
    y_true_cum = tf.cumsum(y_true, axis=1)
    y_pred_cum = tf.cumsum(y_pred, axis=1)
    print("Y_TRUE_CUM", y_true_cum)
    print("Y_PRED_CUM", y_pred_cum)

    # Calculate the profit/loss based on the trading strategy:
    # Profit/loss is computed as the product of the prediction and the actual change.
    # This mimics the gain or loss from going 'long' or 'short' based on the sign and magnitude of the prediction.
    # profit_loss = -(y_pred_cum * y_true_cum) / 100  # Negative because we minimize loss in training
    profit_loss = -(tf.gather(y_pred_cum * y_true_cum, tf.range(y_true.shape[1]), axis=1) / tf.cast(tf.range(1, y_true.shape[1]+1), tf.float32))
    print("PROFIT_LOSS", profit_loss)
    # Regulate profit/loss by the decision to act
    regulated_loss = decision * profit_loss
    print("REGULATED_LOSS", regulated_loss)

    # Penalty for not acting when there is significant market movement
    y_true_reverse_cum = tf.reverse(tf.cumsum(tf.reverse(y_true, [1]), axis=1), [1])
    abstain_penalty = financial_penalty * (1 - decision) * y_true_reverse_cum  # Apply penalty based on reverse cumulative movements missed
    print("ABSTAIN_PENALTY", abstain_penalty)

    prediction_error = tf.square(y_true_cum - y_pred_cum) * decision  # Penalize the model for incorrect predictions

    # Combine into a total loss
    total_loss = tf.reduce_mean(regulated_loss + abstain_penalty + prediction_error)

    return total_loss


def custom_loss2(y_true, y_pred_full, profit_multiplier = 4, financial_penalty = 2, error_weight = 1):
    # Split predictions and decision signals
    y_pred = y_pred_full[:, :, :-1]  # Predicted day-over-day percent changes
    decision = y_pred_full[:, :, -1:]  # Decision output (already sigmoid)
    print("Decision", decision) 

    # Calculate cumulative sums
    # In real trading, we'd consider each day's change as an independent event where the model decides to 'buy' (predict positive) or 'sell' (predict negative).
    y_true_cum = tf.cumsum(y_true, axis=1)
    y_pred_cum = tf.cumsum(y_pred, axis=1)
    print("Y_TRUE_CUM", y_true_cum)
    print("Y_PRED_CUM", y_pred_cum)

    profit = tf.zeros_like(y_true_cum)
    day_numbers = tf.cast(tf.range(1, tf.shape(y_true)[1] + 1), dtype=tf.float32)
    day_numbers = tf.reshape(day_numbers, (1, -1, 1))
    reverse_day_numbers = tf.cast(tf.range(tf.shape(y_true)[1], 0, -1), dtype=tf.float32)
    reverse_day_numbers = tf.reshape(reverse_day_numbers, (1, -1, 1))

    profit = tf.where(
        (y_pred_cum > 0) & (y_true_cum > 0), 
        y_true_cum, 
        profit
    )
    profit = tf.where(
        (y_pred_cum > 0) & (y_true_cum < 0),
        -y_true_cum,
        profit
    )
    profit = tf.where(
        (y_pred_cum < 0) & (y_true_cum > 0),
        -y_true_cum,
        profit
    )
    profit = tf.where(
        (y_pred_cum < 0) & (y_true_cum < 0),
        y_true_cum,
        profit
    )

    print("cumulative profit loss ", profit)

    avg_cumulative_profit_per_day = profit / day_numbers


    profit_loss = -avg_cumulative_profit_per_day * profit_multiplier
    print("PROFIT_LOSS", profit_loss)
    # Regulate profit/loss by the decision to act
    #
    regulated_loss = decision * profit_loss
    print("REGULATED_LOSS", regulated_loss)

    # Penalty for not acting when there is significant market movement
    y_true_reverse_cum = tf.reverse(tf.cumsum(tf.reverse(y_true, [1]), axis=1), [1])
    y_true_reverse_cum_per_day = y_true_reverse_cum / reverse_day_numbers
    abstain_penalty = financial_penalty * (1 - decision) * y_true_reverse_cum_per_day  # Apply penalty based on reverse cumulative movements missed
    print("ABSTAIN_PENALTY", abstain_penalty)

    # add small constant to decision so we evaluate the error a little bit 
    decision = decision + 1e-1

    prediction_error = error_weight *tf.abs(y_true_cum - y_pred_cum) * decision   # Penalize the model for incorrect predictions

    # Combine into a total loss
    total_loss = tf.reduce_mean(regulated_loss + abstain_penalty + prediction_error)

    return total_loss



def custom_loss_updated(y_true, y_pred_full, profit_multiplier=1, financial_penalty=0, error_weight=0, decision_thres=0.5):
    # Split predictions and decision signals
    y_pred = y_pred_full[:, :, :-1]  # Predicted day-over-day percent changes
    decision = y_pred_full[:, :, -1:]  # Decision output (already sigmoid)
    print("Decision", decision)

    # Process decision matrix based on the threshold
    # Set all following values in each sequence to zero if a value falls below the threshold
    cum_min_decision = tf.minimum(tf.cumsum(tf.cast(decision < decision_thres, tf.float32), axis=1), 1)
    decision = decision * (1 - cum_min_decision)
    
    print("Processed Decision", decision)

    # Calculate cumulative sums
    y_true_cum = tf.cumsum(y_true, axis=1)
    y_pred_cum = tf.cumsum(y_pred, axis=1)
    print("Y_TRUE_CUM", y_true_cum)
    print("Y_PRED_CUM", y_pred_cum)

    # Calculate profits and losses
    profit = tf.zeros_like(y_true_cum)
    day_numbers = tf.cast(tf.range(1, tf.shape(y_true)[1] + 1), dtype=tf.float32)
    day_numbers = tf.reshape(day_numbers, (1, -1, 1))
    reverse_day_numbers = tf.cast(tf.range(tf.shape(y_true)[1], 0, -1), dtype=tf.float32)
    reverse_day_numbers = tf.reshape(reverse_day_numbers, (1, -1, 1))

    # Conditions for calculating profit
    profit = tf.where(
        (y_pred_cum > 0) & (y_true_cum > 0), 
        tf.minimum(y_pred_cum, y_true_cum),
        profit
    )
    profit = tf.where(
        (y_pred_cum > 0) & (y_true_cum < 0),
        -tf.abs(y_pred_cum),
        profit
    )
    profit = tf.where(
        (y_pred_cum < 0) & (y_true_cum > 0),
        -tf.abs(y_pred_cum),
        profit
    )
    profit = tf.where(
        (y_pred_cum < 0) & (y_true_cum < 0),
        tf.minimum(tf.abs(y_pred_cum), tf.abs(y_true_cum)),
        profit
    )

    print("cumulative profit loss ", profit)

    avg_cumulative_profit_per_day = profit / day_numbers
    profit_loss = -avg_cumulative_profit_per_day * profit_multiplier
    print("PROFIT_LOSS", profit_loss)

    # Regulate profit/loss by the decision to act
    regulated_loss = decision * profit_loss + 1e-10
    print("REGULATED_LOSS", regulated_loss)

    # Penalty for not acting when there is significant market movement
    y_true_reverse_cum = tf.reverse(tf.cumsum(tf.reverse(y_true, [1]), axis=1), [1])
    y_true_reverse_cum_per_day = y_true_reverse_cum / reverse_day_numbers
    abstain_penalty = financial_penalty * (1 - decision) * tf.abs(y_true_reverse_cum_per_day)
    print("ABSTAIN_PENALTY", abstain_penalty)

    # Compute prediction error
    prediction_error = error_weight * tf.abs(y_true - y_pred)  # Penalize the model for incorrect predictions
    print("PREDICTION_ERROR", prediction_error)

    # Combine into a total loss
    total_loss = tf.reduce_mean(regulated_loss + abstain_penalty + prediction_error)

    return total_loss



def mae_decision_model(y_true, y_pred_full):
    # Split predictions and decision signals
    y_pred = y_pred_full[:, :, :-1]  # Predicted day-over-day percent changes
    decision = y_pred_full[:, :, -1:]  # Decision output (already sigmoid)

    # return mae of y_pred and y_true 
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return mae

