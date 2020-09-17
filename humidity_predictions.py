import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.svm import SVR

pd.options.mode.chained_assignment = None


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def humidity_caller(state, dist):
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = dist + '%2C' + state + '.csv'
    file = file.replace('+', '%2B')
    df = pd.read_csv(base_url + file)
    cols = ['date_time', 'maxtempC', 'mintempC', 'humidity', 'tempC', 'pressure']
    cols1 = ['maxtempC', 'mintempC', 'humidity', 'tempC', 'pressure']
    cols2 = ['year', 'month', 'maxtempC', 'mintempC', 'humidity', 'tempC', 'pressure']
    df['date_time'] = pd.to_datetime(df['date_time'])
    df1 = df[cols]

    df1['month'] = df1['date_time'].apply(lambda mon: mon.strftime('%m'))
    df1['year'] = df1['date_time'].apply(lambda year: year.strftime('%Y'))

    df1.drop(['date_time'], 1, inplace=True)

    g = df1.groupby(['year', 'month'], as_index=False)

    monthly_averages = g.aggregate(np.mean)
    monthly_averages[cols1] = monthly_averages[cols1].astype('int8')

    df2 = monthly_averages

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

    ####################################################################################

    x = []
    t = []
    for i, j in df2.iterrows():
        if int(j[0]) < 2019:
            x.append(j)
        elif int(j[0]) == 2019:
            t.append(j)

    x1 = pd.DataFrame(x, columns=cols2)
    x2 = pd.DataFrame(t, columns=cols2)

    forecast_out = 12

    x1['prediction'] = x1[['humidity']].shift(-forecast_out)

    X = np.array(x1.drop(['prediction', 'year', 'month'], 1))
    X = X[:-forecast_out]

    y = np.array(x1['prediction'])
    y = y[:-forecast_out]

    i = 1
    mini = (np.Infinity, np.Infinity)
    ind = i
    while i <= 3000:
        svm = SVR(kernel='rbf', C=i)
        svm.fit(X, y)

        x_forecast = np.array(x1.drop(['prediction', 'year', 'month'], 1))[-forecast_out:]
        svm_prediction = svm.predict(x_forecast)

        svm_prediction = svm_prediction.astype('int8')
        svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
        x2.index = svm_prediction.index
        svm_prediction['Original'] = x2['humidity']

        svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
        svm_prediction['diff'] = abs(svm_prediction['diff'])

        average_error = svm_prediction['diff'].sum()
        average_error = average_error / forecast_out
        max_difference = svm_prediction['diff'].max()

        if average_error < mini[0]:
            mini = (average_error, max_difference)
            ind = i
        elif average_error == mini[0]:
            if max_difference < mini[1]:
                mini = (average_error, max_difference)
                ind = i
        i += 1

    x = []

    for i, j in df2.iterrows():
        if int(j[0]) <= 2019:
            x.append(j)

    x1 = pd.DataFrame(x, columns=cols2)
    forecast_out = 12
    x1['prediction'] = x1[['humidity']].shift(-forecast_out)

    X = np.array(x1.drop(['prediction', 'year', 'month'], 1))
    X = X[:-forecast_out]

    y = np.array(x1['prediction'])
    y = y[:-forecast_out]

    svm = SVR(kernel='rbf', C=ind)
    svm.fit(X, y)

    x_forecast = np.array(x1.drop(['prediction', 'year', 'month'], 1))[-forecast_out:]
    svm_prediction = svm.predict(x_forecast)

    svm_prediction = svm_prediction.astype('int8')
    svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
    svm_prediction['Month'] = range(1, 13)

    df8 = pd.DataFrame()
    df8['Month'] = range(1, 13)
    df8['SVM'] = svm_prediction['Predicted']
    df8['ANN'] = df4['Predicted']
    df3 = df2[df2['year'] == '2019']
    df3.reset_index(inplace=True, drop=True)
    df8['Original'] = df3['humidity']

    c1 = 0
    c2 = 0

    for i, j in df8.iterrows():
        d1 = abs(int(j[1]) - int(j[3]))
        d2 = abs(int(j[2]) - int(j[3]))

        if d1 < d2:
            c1 += 1
        elif d1 > d2:
            c2 += 1
        else:
            c2 += 1
            c1 += 1

    if c1 > c2:
        model_selected = 1
    else:
        model_selected = 2

    c = []

    for i, j in df8.iterrows():
        d1 = abs(int(j[1]) - int(j[3]))
        d2 = abs(int(j[2]) - int(j[3]))

        if d1 < d2:
            c.append(j[1])
        elif d1 > d2:
            c.append(j[2])
        else:
            if model_selected == 1:
                c.append(j[1])
            else:
                c.append(j[2])

    df5 = pd.DataFrame()
    df5['Predicted'] = c
    df5['Month'] = range(1, 13)

    df5.to_csv(f'outputs/humidity/{dist},{state}.csv', index=False, header=True)
