from keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from enum import Enum
from tensorflow.keras.callbacks import EarlyStopping

class ModelTypes(Enum):
    Traditional = 1
    SAE = 2
    AE = 3 # Traditional autoendoder 
    AN = 4 # Attention network 

class RNNModel:
    def __init__(self,modelType,input_shape,output_shape, num_autoencoder_layers= None, num_encoder_layers = None):
        self.layers = [] 
        self.modelType = modelType
        self.input_shape = input_shape
        self.num_layers = 0
        self.num_autoencoder_layers = num_autoencoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.output_shape = output_shape
    
    def addLSTMLayer(self,units,return_sequences=True,activation='tanh'):
        
        if self.num_layers == 0:
            if self.modelType == ModelTypes.SAE:
                self.layers.append(LSTM(units, input_shape=(None,self.input_shape[2]),return_sequences=return_sequences,activation=activation,name = 'LSTM_1'))
            else:
                self.layers.append(LSTM(units, input_shape=(None,self.input_shape[2]),return_sequences=return_sequences,activation=activation,name = 'LSTM_1'))
        else:
            self.self.layers.append(LSTM(units,return_sequences=return_sequences,activation=activation,name = 'LSTM_'+str(self.num_layers+1)))
        
        self.num_layers += 1
    
    def addGRULayer(self,units,return_sequences=True,activation='tanh'):
        if self.num_layers == 0:
            if self.modelType == ModelTypes.SAE:
                self.layers.append(LSTM(units, input_shape=(None,self.input_shape[2]),return_sequences=return_sequences,activation=activation,name = 'GRU_1'))
            else:
                self.layers.append(LSTM(units, input_shape=(None,self.input_shape[2]),return_sequences=return_sequences,activation=activation,name = 'GRU_1'))
        else:
            self.layers.append(GRU(units,return_sequences=return_sequences,activation=activation,name = 'GRU_'+str(self.num_layers+1)))
        
        self.num_layers += 1
    
    # def addEchoStateLayer(self,units,return_sequences=True,activation='tanh'):
    #     if self.num_layers == 0:
    #         self.model.add(EchoState(units, input_shape=self.input_shape,return_sequences=return_sequences,activation=activation),name = 'EchoState_1')
    #     else:
    #         self.model.add(EchoState(units,return_sequences=return_sequences,activation=activation),name = 'EchoState_'+str(self.num_layers+1))
        
    #     self.num_layers += 1

    def addDropoutLayer(self,rate):
        self.layers.append(Dropout(rate,name = 'Dropout_'+str(self.num_layers)))
    
    def addBatchNormLayer(self):
        self.layers.append(BatchNormalization(name = 'BatchNorm_'+str(self.num_layers)))
    
    def buildAutoencoderBlock(self):
        encoder = Sequential()
        encoder.add(LSTM(self.input_shape[2], input_shape=(self.input_shape[1], self.input_shape[2]), return_sequences=False))
        encoder.add(RepeatVector(self.input_shape[1]))

        decoder = Sequential()
        decoder.add(LSTM(self.input_shape[2], return_sequences=True))

        return Sequential([encoder,decoder])
    

    
    def buildModel(self):
        new_model = Sequential()
        if self.modelType == ModelTypes.Traditional:
            for layer in self.layers:
                new_model.add(layer)
            self.addLSTMLayer(units = 50,return_sequences = False)
            new_model.add(Dense(units = 6))

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
                if next_layer.name.split('_')[0] == 'LSTM' or next_layer.name.split('_')[0] == 'GRU':
                    cur_encoder_layers += 1
                new_model.add(next_layer)
                total_layers += 1

            new_model.add(LSTM(units = 10,return_sequences = False))
            new_model.add(RepeatVector(self.output_shape, name = 'repeat_vector'))
            
            for i in range(total_layers,len(self.layers)):
                new_model.add(self.layers[i])
            
            new_model.add(TimeDistributed(Dense(1), name='time_distributed_output'))

        
        new_model.compile(loss='mae', optimizer=Adam(learning_rate=0.001))
        self.model = new_model
    
    def fit(self,X_train,y_train,validation_data = None,epochs=100,batch_size=32):
                # After building the model
        print(self.model.summary())

        # Before training, inspect the shape of training and validation data
        print("Training data shape:", X_train.shape, y_train.shape)
        print("Validation data shape:", validation_data[0].shape, validation_data[1].shape)

        # Optionally, implement a custom training loop for further debugging
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Metric to monitor (e.g., validation loss)
            patience=15,          # Number of epochs with no improvement before stopping
            restore_best_weights=True  # Restore model weights to the best epoch
        )

        # Train the model with early stopping

        self.model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=validation_data,callbacks=[early_stopping])
    
    def predict(self,X):
        return self.model.predict(X)
    

    # def build_
