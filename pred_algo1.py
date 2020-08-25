import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
from sklearn.svm import SVR

pd.options.mode.chained_assignment = None

base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'


class LinearRegressionModel(nn.Module, ABC):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def pre_processing(df, cols):
    cols1 = copy.deepcopy(cols)
    for k in cols1:
        del k[0]
    cols2 = copy.deepcopy(cols1)
    for k in cols2:
        k.insert(0, 'year')
        k.insert(1, 'month')

    df['date_time'] = pd.to_datetime(df['date_time'])
    df1 = df[cols]

    df1['month'] = df1['date_time'].apply(lambda mon: mon.strftime('%m'))
    df1['year'] = df1['date_time'].apply(lambda year: year.strftime('%Y'))

    df1.drop(['date_time'], 1, inplace=True)

    g = df1.groupby(['year', 'month'], as_index=False)

    monthly_averages = g.aggregate(np.mean)
    monthly_averages[cols1] = monthly_averages[cols1].astype('int8')

    return monthly_averages


def ANN(state, dist, cols):
    file = dist + '%2C' + state + '.csv'
    file = file.replace('+', '%2B')
    df = pd.read_csv(base_url + file)

    cols1 = copy.deepcopy(cols)
    for k in cols1:
        del k[0]
    cols2 = copy.deepcopy(cols1)
    for k in cols2:
        k.insert(0, 'year')
        k.insert(1, 'month')

    df2 = pre_processing(df, cols)

    x = []
    t = []
    for i, j in df2.iterrows():
        if int(j[0]) <= 2019:
            x.append(j)

    x1 = pd.DataFrame(x, columns=cols2)

    forecast_out = 12
    x1['prediction'] = x1[['humidity']].shift(-forecast_out)

    X = np.array(x1.drop(['prediction', 'year', 'month'], 1))
    X = X[:-forecast_out]

    y = np.array(x1[['prediction']])
    y = y[:-forecast_out]

    x_forecast = np.array(x1.drop(['prediction', 'year', 'month'], 1))[-forecast_out:]

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    model = LinearRegressionModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    learning_rate = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 2000
    for epoch in range(epochs):
        epoch += 1
        inputs = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(y.astype(np.float32))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    predicted_new = model(torch.FloatTensor(x_forecast)).data.numpy()
    predicted_new = predicted_new.reshape(12, )

    df4 = pd.DataFrame()
    df4['Predicted'] = predicted_new
    df4['Month'] = range(1, 13)
    df4 = df4.astype('int8')

    return df4
