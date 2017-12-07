'''
Created on 30 nov 2017

@author: mantica
'''
# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import metrics
import keras.backend as K


class ModelUtils(object):
    '''
    classdocs
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        
    @staticmethod    
    def r2_keras(y_true, y_pred):
        """
        Coefficient of Determination
        """    
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    @staticmethod
    def buildLstmInput(dataset, norm_cols, window_len):
        """
        Model output is next price normalised to 10th previous closing price
        Return array of sequences
        """
        # array of sequences    
        LSTM_inputs = []
        for i in range(len(dataset) - window_len):
            # Get a windows of rows
            temp_set = dataset[i:(i + window_len)].copy()
            # Normalize from -1 to 1
            for col in norm_cols:
                temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
            LSTM_inputs.append(temp_set)
        
        return LSTM_inputs
    
    @staticmethod
    def buildLstmOutput(dataset, target, window_len):
        """
        Model output is next price normalised to 10th previous closing price
        """    
        return (dataset[target][window_len:].values / dataset[target][:-window_len].values) - 1    
    
    @staticmethod
    def build_model(inputs, output_size, neurons_lv1, neurons_lv2, activ_func="linear",
                    dropout=0.3, loss="mean_squared_error", optimizer="adam"):
        """
        LSTM
        """
        model = Sequential()
        
        model.add(LSTM(
             input_shape=(inputs.shape[1], inputs.shape[2]),
             units=neurons_lv1,
             return_sequences=True))
        
        model.add(Dropout(dropout))
        
        model.add(LSTM(
              units=neurons_lv2,
              return_sequences=False))
        
        model.add(Dropout(dropout))
        
        model.add(Dense(units=output_size))
        
        model.add(Activation(activ_func))
        
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics.mse, metrics.mae, ModelUtils.r2_keras])
        # Summarize model
        print(model.summary())
        return model
