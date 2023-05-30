# Ozge kabak
# 17/02/2023
# main function to run LSTM for single parameter under one layer


# Importing Libraries
import random
import numpy as np

from math import sqrt
from numpy import concatenate
import tensorflow as tf
from keras.losses import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from attention import Attention
from keras.layers import Dense, Activation, LSTM, Bidirectional, Conv1D, \
    GRU, SimpleRNN
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from sklearn.preprocessing import StandardScaler

from dataProcess import dataProcess


def mape(actual, predicted) -> float:
    # Convert actual and predicted
    # to numpy array data type if not already
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual),
        np.array(predicted)

    # Calculate the MAPE value and return
    return round(np.mean(np.abs((actual - predicted) / actual)) * 100, 2)


def train_model(seed_num, epoch, modelType, testFlag, patientFlag, layerNumber, featureNumber, plotFlag,attentionFlag):
    os.environ['PYTHONHASHSEED'] = str(seed_num)

    random.seed(seed_num)
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)

    train_X0, train_y0, test_X0, test_y0, scaler = dataProcess(featureNumber, patientFlag, plotFlag)

    model = Sequential()
    if attentionFlag==0:
        if layerNumber == 1:
            if modelType == 0:
                model.add(SimpleRNN(1024, return_sequences=False, input_shape=(train_X0.shape[1], train_X0.shape[2])))

            elif modelType == 1:
                model.add(LSTM(1024, return_sequences=False, input_shape=(train_X0.shape[1], train_X0.shape[2])))

            elif modelType == 2:
                model.add(GRU(1024, return_sequences=False, input_shape=(train_X0.shape[1], train_X0.shape[2])))

            elif modelType == 3:
                model.add(Bidirectional(
                    SimpleRNN(1024, return_sequences=False, input_shape=(train_X0.shape[1], train_X0.shape[2]))))

            elif modelType == 4:
                model.add(
                    Bidirectional(LSTM(1024, return_sequences=False, input_shape=(train_X0.shape[1], train_X0.shape[2]))))

            elif modelType == 5:
                model.add(
                    Bidirectional(GRU(1024, return_sequences=False, input_shape=(train_X0.shape[1], train_X0.shape[2]))))

            elif modelType == 6:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(SimpleRNN(1024, return_sequences=False))

            elif modelType == 7:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(LSTM(1024, return_sequences=False))

            elif modelType == 8:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(GRU(1024, return_sequences=False))

            model.add(Dense(1024))
            model.add(Dense(512))

        elif layerNumber == 2:
            if modelType == 0:

                model.add(SimpleRNN(512, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(SimpleRNN(512, return_sequences=False))

            elif modelType == 1:
                model.add(LSTM(512, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(LSTM(512, return_sequences=False))

            elif modelType == 2:
                model.add(GRU(512, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(GRU(512, return_sequences=False))

            elif modelType == 3:
                model.add(Bidirectional(SimpleRNN(512, return_sequences=True, input_shape=(train_X0.shape[1],
                                                                                           train_X0.shape[2]))))
                model.add(Bidirectional(SimpleRNN(512, return_sequences=False)))

            elif modelType == 4:
                model.add(Bidirectional(LSTM(512, return_sequences=True, input_shape=(train_X0.shape[1],
                                                                                      train_X0.shape[2]))))
                model.add(Bidirectional(LSTM(512, return_sequences=False)))

            elif modelType == 5:
                model.add(Bidirectional(GRU(512, return_sequences=True, input_shape=(train_X0.shape[1],
                                                                                     train_X0.shape[2]))))
                model.add(Bidirectional(GRU(512, return_sequences=False)))

            elif modelType == 6:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(SimpleRNN(512, return_sequences=True))
                model.add(SimpleRNN(512, return_sequences=False))

            elif modelType == 7:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(LSTM(512, return_sequences=True))
                model.add(LSTM(512, return_sequences=False))

            elif modelType == 8:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(GRU(512, return_sequences=True))
                model.add(GRU(512, return_sequences=False))
    elif attentionFlag==1:
        if layerNumber == 1:
            if modelType == 0:
                model.add(SimpleRNN(1024, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))

            elif modelType == 1:
                model.add(LSTM(1024, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))

            elif modelType == 2:
                model.add(GRU(1024, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))

            elif modelType == 3:
                model.add(Bidirectional(
                    SimpleRNN(1024, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2]))))

            elif modelType == 4:
                model.add(
                    Bidirectional(
                        LSTM(1024, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2]))))

            elif modelType == 5:
                model.add(
                    Bidirectional(
                        GRU(1024, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2]))))

            elif modelType == 6:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(SimpleRNN(1024, return_sequences=True))

            elif modelType == 7:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(LSTM(1024, return_sequences=True))

            elif modelType == 8:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(GRU(1024, return_sequences=True))

            model.add(Attention(512))
            model.add(Dense(1024))
            model.add(Dense(512))

        elif layerNumber == 2:
            if modelType == 0:

                model.add(SimpleRNN(512, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(SimpleRNN(512, return_sequences=True))

            elif modelType == 1:
                model.add(LSTM(512, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(LSTM(512, return_sequences=True))

            elif modelType == 2:
                model.add(GRU(512, return_sequences=True, input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(GRU(512, return_sequences=True))

            elif modelType == 3:
                model.add(Bidirectional(SimpleRNN(512, return_sequences=True, input_shape=(train_X0.shape[1],
                                                                                           train_X0.shape[2]))))
                model.add(Bidirectional(SimpleRNN(512, return_sequences=True)))

            elif modelType == 4:
                model.add(Bidirectional(LSTM(512, return_sequences=True, input_shape=(train_X0.shape[1],
                                                                                      train_X0.shape[2]))))
                model.add(Bidirectional(LSTM(512, return_sequences=True)))

            elif modelType == 5:
                model.add(Bidirectional(GRU(512, return_sequences=True, input_shape=(train_X0.shape[1],
                                                                                     train_X0.shape[2]))))
                model.add(Bidirectional(GRU(512, return_sequences=True)))

            elif modelType == 6:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(SimpleRNN(512, return_sequences=True))
                model.add(SimpleRNN(512, return_sequences=True))

            elif modelType == 7:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(LSTM(512, return_sequences=True))
                model.add(LSTM(512, return_sequences=True))

            elif modelType == 8:
                model.add(Conv1D(filters=32, kernel_size=1, activation='relu',
                                 input_shape=(train_X0.shape[1], train_X0.shape[2])))
                model.add(GRU(512, return_sequences=True))
                model.add(GRU(512, return_sequences=True))

            model.add(Attention(512))


    model.add(Dense(6))
    model.add(Activation("linear"))
    rmsprop = optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-08)
    model.compile(loss='mse',
                  optimizer=rmsprop, metrics=['accuracy'])

    history = model.fit(train_X0, train_y0, epochs=epoch, batch_size=32)
    # model.summary()

    # plot history
    if plotFlag == 1:
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title("Loss in LSTM Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if testFlag == 1:
        # make a prediction
        yhat = model.predict(test_X0)
        test_X1 = test_X0.reshape((test_X0.shape[0], test_X0.shape[2]))

        # invert scaling for forecast
        # inv_yhat = concatenate((yhat, test_X[:, 0:]), axis=1)
        inv_yhat = concatenate((test_X1, yhat), axis=1)

        inv_yhat1 = scaler.inverse_transform(inv_yhat)
        inv_yhat1 = inv_yhat1[:, -6:]

        # invert scaling for actual
        inv_y = concatenate((test_X1, test_y0), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y1 = inv_y[:, -6:]

        # calculate RMSE
        rmse_val = sqrt(mean_squared_error(inv_y1, inv_yhat1))
        rmse_test = sqrt(mean_squared_error(inv_y1, inv_yhat1))
        print(rmse_test)
        mae = mean_absolute_error(inv_y1, inv_yhat1)
        print(mae)
        mape_er = mape(inv_y1, inv_yhat1)
        print(mape_er)
        r = r2_score(inv_y1, inv_yhat1)
        print(r)

    return rmse_val, rmse_test, mae, mape_er, r
