from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from tensorflow.keras.layers import GRU, LSTM, RNN, Dense, Dropout, GRUCell, LSTMCell
from tensorflow.keras import Model, Input

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LayerNormalization

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
    def __init__(self, rnn_type, rnn_size, rnn_dropout=0, dense_size=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.rnn_type = rnn_type
        if rnn_type.lower() == "gru":
            self.rnn = GRU(
                units=rnn_size, activation="tanh", return_state=True, return_sequences=True, dropout=rnn_dropout
            )
            self.dense_state = Dense(dense_size, activation="tanh")  # For projecting GRU state
        elif rnn_type.lower() == "lstm":
            self.rnn = LSTM(
                units=rnn_size,
                activation="tanh",
                return_state=True,
                return_sequences=True,
                dropout=rnn_dropout,
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
        self.encoder1 = Encoder(rnn_type="lstm", rnn_size=128, rnn_dropout=0.3, dense_size=64)
        self.encoder2 = Encoder(rnn_type="lstm", rnn_size=64, rnn_dropout=0.2, dense_size=64)
        self.decoder = Decoder1(
            rnn_type="lstm",
            rnn_size=64,
            predict_sequence_length=predict_sequence_length,
            use_attention=True,
            attention_sizes=32,
            attention_heads=1,
            attention_dropout=.1
        )

    def call(self, inputs, training=False):
        encoder_output1, encoder_state1 = self.encoder1(inputs)

        encoder_output2, encoder_state2 = self.encoder2(encoder_output1)
        decoder_init_input = tf.zeros_like(inputs[:, 0, :1])  # Example for initialization
        
        
        # Explicitly call the decoder with keyword arguments
        decoder_output, attention_weights = self.decoder(
            decoder_features=None,
            decoder_init_input=decoder_init_input,
            init_state=encoder_state2,
            encoder_output=encoder_output2,  # Passed explicitly as per the updated signature
            teacher=None,
            scheduler_sampling=0,
            training=training
        )
        
        if training:
            # During training, return only the decoder output.
            return decoder_output
        else:
            # During evaluation or inference, return both output and attention weights.
            print("Attention Weights Shape: ", attention_weights.shape)
            return decoder_output, attention_weights


 



