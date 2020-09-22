from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model


def get_model17(height, n_feature):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(height, n_feature),
                     kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(height, n_feature),
                     kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(height, n_feature),
                     kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    model.add(LSTM(10, return_sequences=True, kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    model.add(Dropout(0.2))
    model.add(
        Bidirectional(LSTM(20, return_sequences=True, kernel_initializer='glorot_normal', bias_initializer='Zeros')))
    model.add(Dropout(0.2))
    model.add(LSTM(30, kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu', kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu', kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    return model