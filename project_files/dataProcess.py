
import pandas as pd

from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import StandardScaler

def dataProcess(featureNumber,patientFlag, plotFlag):

    ####################################################################################################################
    # Initialization
    patientTrainList = ['1/glucose.csv', '2/glucose.csv', '3/glucose.csv',
                        '4/glucose.csv', '5/glucose.csv', '6/glucose.csv',
                        '7/glucose.csv', '8/glucose.csv', '9/glucose.csv']

    ####################################################################################################################


    dataset = pd.read_csv(patientTrainList[patientFlag], header=0, index_col=0, usecols=[i for i in range(featureNumber + 1)])


    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in - 1, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out + 1):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    df5 = series_to_supervised(dataset, 6, 6)

    # ensure all data is float
    valuesTrain = df5.values

    # convert elements of valuesTrain from mmol/L to mg/dL
    for i in range(0, len(valuesTrain)):
        for j in range(0, len(valuesTrain[0])):
            valuesTrain[i][j] = valuesTrain[i][j] * 18.0182

    valuesTrain = valuesTrain.astype('float32')
    split_v = round(len(valuesTrain) * 0.80)
    test_values = valuesTrain[split_v:,:]
    # normalize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(valuesTrain)
    test_scaled = scaler.fit_transform(test_values)


    train=scaled
    test = test_scaled


    # split into input and outputs
    train_X0, train_y0 = train[:split_v, :-6], train[:split_v, -6:]
    test_X0, test_y0 = test[:, :-6], test[:, -6:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X0 = train_X0.reshape((train_X0.shape[0], 1, train_X0.shape[1]))
    test_X0= test_X0.reshape((test_X0.shape[0], 1, test_X0.shape[1]))

    return  train_X0, train_y0, test_X0, test_y0, scaler