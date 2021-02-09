import enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

pd.options.mode.chained_assignment = None


class Features(enum.Enum):
    f1 = ['Rain']
    f2 = ['Rain', 'Temp']
    f3 = ['Rain', 'Humidity']
    f4 = ['Rain', 'Temp', 'Humidity']


def negative_checker(s):
    if s < 0:
        return 0
    else:
        return s


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def merge_rain_and_temp(_x1, _t1):
    rain_years = list(_x1['Year'])
    temp_years = list(_t1['Year'])
    diff_years = list(list(set(rain_years) - set(temp_years)) + list(set(temp_years) - set(rain_years)))
    actual_years = list(set(rain_years) - set(diff_years))
    _r = []
    for year in actual_years:
        for _i, _j in _t1.iterrows():
            if _j[0] == year:
                _r.append(_j[1])

    return _r


def find_second_last_year(_r):
    years = list(_r['Year'])
    if years[-1] == '2019':
        return years[-3]
    else:
        return years[-2]


def find_last_year(_r):
    years = list(_r['Year'])
    if years[-1] == '2019':
        return years[-2]
    else:
        return years[-1]


def rain_caller(state, dist):
    dist1 = dist.replace('+', ' ')
    state1 = state.replace('+', ' ')

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file1 = 'rainfall_data_3.csv'

    rain = pd.read_csv(base_url + file1)
    rain1 = rain[rain['State'] == state1]
    rain1 = rain1[rain1['District'] == dist1]
    rain1.reset_index(drop=True, inplace=True)

    file2 = 'whole_temp_2.csv'

    temp = pd.read_csv(base_url + file2)
    temp1 = temp[temp['State'] == state1]
    temp1 = temp1[temp1['District'] == dist1]
    temp1.reset_index(drop=True, inplace=True)

    file3 = dist + '%2C' + state + '.csv'
    file3 = file3.replace('+', '%2B')

    humidity = pd.read_csv(base_url + file3)

    h1 = humidity[['date_time', 'humidity']]
    h1['date_time'] = pd.to_datetime(h1['date_time'])

    h1['month'] = h1['date_time'].apply(lambda mon: mon.strftime('%m'))
    h1['year'] = h1['date_time'].apply(lambda year: year.strftime('%Y'))

    h1.drop(['date_time'], 1, inplace=True)

    g = h1.groupby(['year', 'month'], as_index=False)

    monthly_averages = g.aggregate(np.mean)
    monthly_averages[['humidity']] = monthly_averages[['humidity']].astype('int8')
    h2 = monthly_averages

    last_year = int(find_last_year(rain1))
    second_last_year = int(find_second_last_year(rain1))

    humidity_2018 = []
    k = 1
    for i, j in h2.iterrows():
        if int(j[0]) == last_year:
            if int(j[1]) == k:
                humidity_2018.append(int(j[2]))
                k += 1

    temp_2018 = []

    temp_2018_df = temp1[temp1['Year'] == last_year]
    for i in range(3, 15):
        temp_2018.append(round(temp_2018_df.iloc[0, i], 2))

    del temp_2018_df

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    working_list = []

    def SVM1_2018(features, local_month, local_month_loc):
        if features == ['Rain']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / forecast_out
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []
            tt = []

            for i, j in dft.iterrows():
                if int(j[0]) < second_last_year:
                    xt.append(j)
                if int(j[0]) == second_last_year:
                    tt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / forecast_out
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / forecast_out
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []
            tt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    xt.append(j)
                if int(j[0]) == second_last_year:
                    tt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            xt2 = pd.DataFrame(tt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            x2.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)
            xt2.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / forecast_out
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)

        return svm_prediction[-1]

    def MLR1_2018(features, local_month, local_month_loc):
        if features == ['Rain']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        return lr_forecast[-1]

    def ANN1_2018(features, local_month, local_month_loc):
        if features == ['Rain']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1[['prediction']])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(forecast_out, )

        elif features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1[['prediction']])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(forecast_out, )

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1[['prediction']])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(forecast_out, )

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1[['prediction']])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(forecast_out, )

        return predicted_new[-1]

    def SVM2_2018(features, local_month, local_month_loc):
        if features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []
            tt = []

            for i, j in dft.iterrows():
                if int(j[0]) < second_last_year:
                    xt.append(j)
                if int(j[0]) == second_last_year:
                    tt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            xt2 = pd.DataFrame(tt, columns=dft.columns)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])
            if x2.shape[0] == xt2.shape[0]:
                x2['Temp'] = xt2[local_month]
            else:
                x2['Temp'] = merge_rain_and_temp(x2, xt2[['Year', local_month]])

            X = np.array(x1[['Temp']])

            y = np.array(x1[local_month])

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x2[['Temp']])
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / 1
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            X = np.array(x1[['Temp']])

            y = np.array(x1[local_month])

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array([[temp_2018[local_month_loc]]])
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            yrs1 = x2['Year'].to_list()

            y1 = []
            for yr in yrs1:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y1.append(h3.iloc[local_month_loc, 2])

            x2['Humidity'] = y1

            X = np.array(x1[['Humidity']])

            y = np.array(x1[local_month])

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x2[['Humidity']])
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / 1
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Humidity']])

            y = np.array(x1[local_month])

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array([[humidity_2018[local_month_loc]]])
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []
            tt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    xt.append(j)
                if int(j[0]) == second_last_year:
                    tt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            xt2 = pd.DataFrame(tt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            x2.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)
            xt2.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])
            if x2.shape[0] == xt2.shape[0]:
                x2['Temp'] = xt2[local_month]
            else:
                x2['Temp'] = merge_rain_and_temp(x2, xt2[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            yrs1 = x2['Year'].to_list()

            y1 = []
            for yr in yrs1:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y1.append(h3.iloc[local_month_loc, 2])

            x2['Humidity'] = y1

            X = np.array(x1[['Temp', 'Humidity']])

            y = np.array(x1[local_month])

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x2[['Temp', 'Humidity']])
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / 1
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Temp', 'Humidity']])

            y = np.array(x1[local_month])

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array([[temp_2018[local_month_loc], humidity_2018[local_month_loc]]])
            svm_prediction = svm.predict(x_forecast)

        return svm_prediction[-1]

    def MLR2_2018(features, local_month, local_month_loc):
        if features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            X = np.array(x1[['Temp']])

            y = np.array(x1[local_month])

            x_forecast = np.array([[temp_2018[local_month_loc]]])

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Humidity']])

            y = np.array(x1[local_month])

            x_forecast = np.array([[humidity_2018[local_month_loc]]])

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Temp', 'Humidity']])

            y = np.array(x1[local_month])

            x_forecast = np.array([[temp_2018[local_month_loc], humidity_2018[local_month_loc]]])

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        return lr_forecast[-1]

    def ANN2_2018(features, local_month, local_month_loc):
        if features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            X = np.array(x1[['Temp']])

            y = np.array(x1[[local_month]])

            x_forecast = np.array([[temp_2018[local_month_loc]]])
            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(1, )

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Humidity']])

            y = np.array(x1[[local_month]])

            x_forecast = np.array([[humidity_2018[local_month_loc]]])

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(1, )

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Temp', 'Humidity']])

            y = np.array(x1[[local_month]])

            x_forecast = np.array([[temp_2018[local_month_loc], humidity_2018[local_month_loc]]])

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(1, )

        return predicted_new[-1]

    cols1 = ['SVM-1', 'MLR-1', 'ANN-1', 'SVM-2', 'MLR-2', 'ANN-2', 'SVM-3', 'MLR-3', 'ANN-3', 'SVM-4', 'MLR-4', 'ANN-4']
    cols2 = ['SVM-2', 'MLR-2', 'ANN-2', 'SVM-3', 'MLR-3', 'ANN-3', 'SVM-4', 'MLR-4', 'ANN-4']

    no_temp_values = [[0] * 12] * 12
    temp_values = [[0] * 9] * 12
    no_temp_values1 = pd.DataFrame(no_temp_values, columns=cols1)
    temp_values1 = pd.DataFrame(temp_values, columns=cols2)

    for month_loc, month in enumerate(months):
        k = 0
        for i in Features:
            no_temp_values1.iloc[month_loc, k] = SVM1_2018(i.value, month, month_loc)
            no_temp_values1.iloc[month_loc, k + 1] = MLR1_2018(i.value, month, month_loc)
            no_temp_values1.iloc[month_loc, k + 2] = ANN1_2018(i.value, month, month_loc)

            k += 3

    no_temp_values1['Month'] = months
    no_temp_values1.set_index(['Month'], inplace=True)
    for i in cols1:
        no_temp_values1[i] = no_temp_values1[i].apply(lambda c: negative_checker(c))
        no_temp_values1[i] = no_temp_values1[i].apply(lambda c: round(c, 2))

    for month_loc, month in enumerate(months):
        k = 0
        for i in Features:
            if i.value != ['Rain']:
                temp_values1.iloc[month_loc, k] = SVM2_2018(i.value, month, month_loc)
                temp_values1.iloc[month_loc, k + 1] = MLR2_2018(i.value, month, month_loc)
                temp_values1.iloc[month_loc, k + 2] = ANN2_2018(i.value, month, month_loc)
                k += 3

    temp_values1['Month'] = months
    temp_values1.set_index(['Month'], inplace=True)
    for i in cols2:
        temp_values1[i] = temp_values1[i].apply(lambda c: negative_checker(c))
        temp_values1[i] = temp_values1[i].apply(lambda c: round(c, 2))

    rain_2018 = []

    rain_2018_df = rain1[rain1['Year'] == 2018]
    for i in range(3, 15):
        rain_2018.append(round(rain_2018_df.iloc[0, i], 3))
    del rain_2018_df

    temp_values1['Original'] = rain_2018
    no_temp_values1['Original'] = rain_2018

    temp_values1.to_csv(f'outputs/rainfall/auxiliary/{dist},{state}_temp.csv')
    no_temp_values1.to_csv(f'outputs/rainfall/auxiliary/{dist},{state}_no_temp.csv')

    m1 = [np.Inf] * 12
    m2 = [np.Inf] * 12
    m1_i = [''] * 12
    m2_i = [''] * 12

    for month_loc, month in enumerate(months):
        for col in cols1:
            d = abs(no_temp_values1.loc[month, col] - no_temp_values1.loc[month, 'Original'])
            if d < m1[month_loc]:
                m1[month_loc] = d
                m1_i[month_loc] = col
        for col in cols2:
            d = abs(temp_values1.loc[month, col] - temp_values1.loc[month, 'Original'])
            if d < m2[month_loc]:
                m2[month_loc] = d
                m2_i[month_loc] = col

    for i in range(12):
        if m1[i] < m2[i]:
            temp_list = []
            s = m1_i[i].split('-')
            temp_list.append(int(s[1]))
            temp_list.append(s[0])
            temp_list.append('No Temp')
        elif m1[i] > m2[i]:
            temp_list = []
            s = m2_i[i].split('-')
            temp_list.append(int(s[1]))
            temp_list.append(s[0])
            temp_list.append('Temp')
        else:
            temp_list = []
            s = m2_i[i].split('-')
            temp_list.append(int(s[1]))
            temp_list.append(s[0])
            temp_list.append('Temp')
        working_list.append(temp_list)

    humidity_2020_df = pd.read_csv(f'outputs/humidity/{dist},{state}.csv')
    humidity_2019 = humidity_2020_df['Predicted'].to_list()
    del humidity_2020_df

    temp_2020_df = pd.read_csv(f'outputs/temp/{dist},{state}.csv')
    temp_2019 = temp_2020_df['Predicted'].to_list()
    del temp_2020_df

    def SVM1(features, local_month, local_month_loc):
        if features == ['Rain']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / forecast_out
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            forecast_out = 2
            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []
            tt = []

            for i, j in dft.iterrows():
                if int(j[0]) < second_last_year:
                    xt.append(j)
                if int(j[0]) == second_last_year:
                    tt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / forecast_out
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            forecast_out = 2
            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / forecast_out
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []
            tt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    xt.append(j)
                if int(j[0]) == second_last_year:
                    tt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            xt2 = pd.DataFrame(tt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            x2.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)
            xt2.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 1

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / forecast_out
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)

        return svm_prediction[-1]

    def MLR1(features, local_month, local_month_loc):
        if features == ['Rain']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1['prediction'])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        return lr_forecast[-1]

    def ANN1(features, local_month, local_month_loc):
        if features == ['Rain']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1[['prediction']])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(forecast_out, )

        elif features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1[['prediction']])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(forecast_out, )

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1[['prediction']])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(forecast_out, )

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            forecast_out = 2

            x1['prediction'] = x1[[local_month]].shift(-forecast_out)

            X = np.array(x1.drop(['prediction', 'Year'], 1))
            X = X[:-forecast_out]

            y = np.array(x1[['prediction']])
            y = y[:-forecast_out]

            x_forecast = np.array(x1.drop(['prediction', 'Year'], 1))[-forecast_out:]

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(forecast_out, )

        return predicted_new[-1]

    def SVM2(features, local_month, local_month_loc):
        if features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []
            tt = []

            for i, j in dft.iterrows():
                if int(j[0]) < second_last_year:
                    xt.append(j)
                if int(j[0]) == second_last_year:
                    tt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            xt2 = pd.DataFrame(tt, columns=dft.columns)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])
            if x2.shape[0] == xt2.shape[0]:
                x2['Temp'] = xt2[local_month]
            else:
                x2['Temp'] = merge_rain_and_temp(x2, xt2[['Year', local_month]])

            X = np.array(x1[['Temp']])

            y = np.array(x1[local_month])

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x2[['Temp']])
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / 1
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            X = np.array(x1[['Temp']])

            y = np.array(x1[local_month])

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array([[temp_2019[local_month_loc]]])
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            yrs1 = x2['Year'].to_list()

            y1 = []
            for yr in yrs1:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y1.append(h3.iloc[local_month_loc, 2])

            x2['Humidity'] = y1

            X = np.array(x1[['Humidity']])

            y = np.array(x1[local_month])

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x2[['Humidity']])
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / 1
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Humidity']])

            y = np.array(x1[local_month])

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array([[humidity_2019[local_month_loc]]])
            svm_prediction = svm.predict(x_forecast)

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []
            t = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    x.append(j)
                if int(j[0]) == second_last_year:
                    t.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)
            x2 = pd.DataFrame(t, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []
            tt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) < second_last_year:
                    xt.append(j)
                if int(j[0]) == second_last_year:
                    tt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            xt2 = pd.DataFrame(tt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            x2.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)
            xt2.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])
            if x2.shape[0] == xt2.shape[0]:
                x2['Temp'] = xt2[local_month]
            else:
                x2['Temp'] = merge_rain_and_temp(x2, xt2[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            yrs1 = x2['Year'].to_list()

            y1 = []
            for yr in yrs1:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y1.append(h3.iloc[local_month_loc, 2])

            x2['Humidity'] = y1

            X = np.array(x1[['Temp', 'Humidity']])

            y = np.array(x1[local_month])

            i = 300
            mini = (np.Infinity, np.Infinity)
            ind = i
            while i <= 3000:
                svm = SVR(kernel='rbf', C=i)
                svm.fit(X, y)

                x_forecast = np.array(x2[['Temp', 'Humidity']])
                svm_prediction = svm.predict(x_forecast)

                svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
                svm_prediction['Predicted'] = svm_prediction['Predicted'].apply(lambda x: negative_checker(x))
                x2.index = svm_prediction.index
                svm_prediction['Original'] = x2[local_month]

                svm_prediction['diff'] = svm_prediction['Predicted'] - svm_prediction['Original']
                svm_prediction['diff'] = abs(svm_prediction['diff'])

                average_error = svm_prediction['diff'].sum()
                max_difference = svm_prediction['diff'].max()
                average_error = average_error / 1
                if average_error < mini[0]:
                    mini = (average_error, max_difference)
                    ind = i
                elif average_error == mini[0]:
                    if max_difference < mini[1]:
                        mini = (average_error, max_difference)
                        ind = i
                i += 1
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Temp', 'Humidity']])

            y = np.array(x1[local_month])

            svm = SVR(kernel='rbf', C=ind)
            svm.fit(X, y)

            x_forecast = np.array([[temp_2019[local_month_loc], humidity_2019[local_month_loc]]])
            svm_prediction = svm.predict(x_forecast)

        return svm_prediction[-1]

    def MLR2(features, local_month, local_month_loc):
        if features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            X = np.array(x1[['Temp']])

            y = np.array(x1[local_month])

            x_forecast = np.array([[temp_2019[local_month_loc]]])

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Humidity']])

            y = np.array(x1[local_month])

            x_forecast = np.array([[humidity_2019[local_month_loc]]])

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Temp', 'Humidity']])

            y = np.array(x1[local_month])

            x_forecast = np.array([[temp_2019[local_month_loc], humidity_2019[local_month_loc]]])

            lr = LinearRegression()
            lr.fit(X, y)

            lr_forecast = lr.predict(x_forecast)

        return lr_forecast[-1]

    def ANN2(features, local_month, local_month_loc):
        if features == ['Rain', 'Temp']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)
            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            X = np.array(x1[['Temp']])

            y = np.array(x1[[local_month]])

            x_forecast = np.array([[temp_2019[local_month_loc]]])
            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(1, )

        elif features == ['Rain', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Humidity']])

            y = np.array(x1[[local_month]])

            x_forecast = np.array([[humidity_2019[local_month_loc]]])

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(1, )

        elif features == ['Rain', 'Temp', 'Humidity']:
            df = rain1[['Year', local_month]]
            x = []

            for i, j in df.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    x.append(j)

            x1 = pd.DataFrame(x, columns=df.columns)

            dft = temp1[['Year', local_month]]
            xt = []

            for i, j in dft.iterrows():
                if 2009 <= int(j[0]) <= second_last_year:
                    xt.append(j)

            xt1 = pd.DataFrame(xt, columns=dft.columns)

            x1.reset_index(drop=True, inplace=True)
            xt1.reset_index(drop=True, inplace=True)

            if x1.shape[0] == xt1.shape[0]:
                x1['Temp'] = xt1[local_month]
            else:
                x1['Temp'] = merge_rain_and_temp(x1, xt1[['Year', local_month]])

            yrs = x1['Year'].to_list()

            y = []
            for yr in yrs:
                h3 = h2[h2['year'] == str(int(yr))]
                h3.reset_index(drop=True, inplace=True)
                y.append(h3.iloc[local_month_loc, 2])

            x1['Humidity'] = y

            X = np.array(x1[['Temp', 'Humidity']])

            y = np.array(x1[[local_month]])

            x_forecast = np.array([[temp_2019[local_month_loc], humidity_2019[local_month_loc]]])

            input_dim = X.shape[1]
            output_dim = y.shape[1]

            model = LinearRegressionModel(input_dim, output_dim)

            criterion = nn.MSELoss()

            learning_rate = 0.00001

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            epochs = 20

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
            predicted_new = predicted_new.reshape(1, )

        return predicted_new[-1]

    values = []
    for i, j in zip(months, working_list):
        month_loc = months.index(i)
        if j[2] == 'No Temp':
            if j[1] == 'SVM':
                if j[0] == 1:
                    values.append(SVM1(Features.f1.value, i, month_loc))
                elif j[0] == 2:
                    values.append(SVM1(Features.f2.value, i, month_loc))
                elif j[0] == 3:
                    values.append(SVM1(Features.f3.value, i, month_loc))
                elif j[0] == 4:
                    values.append(SVM1(Features.f4.value, i, month_loc))
            if j[1] == 'MLR':
                if j[0] == 1:
                    values.append(MLR1(Features.f1.value, i, month_loc))
                elif j[0] == 2:
                    values.append(MLR1(Features.f2.value, i, month_loc))
                elif j[0] == 3:
                    values.append(MLR1(Features.f3.value, i, month_loc))
                elif j[0] == 4:
                    values.append(MLR1(Features.f4.value, i, month_loc))
            if j[1] == 'ANN':
                if j[0] == 1:
                    values.append(ANN1(Features.f1.value, i, month_loc))
                elif j[0] == 2:
                    values.append(ANN1(Features.f2.value, i, month_loc))
                elif j[0] == 3:
                    values.append(ANN1(Features.f3.value, i, month_loc))
                elif j[0] == 4:
                    values.append(ANN1(Features.f4.value, i, month_loc))
        else:
            if j[1] == 'SVM':
                if j[0] == 1:
                    values.append(SVM2(Features.f1.value, i, month_loc))
                elif j[0] == 2:
                    values.append(SVM2(Features.f2.value, i, month_loc))
                elif j[0] == 3:
                    values.append(SVM2(Features.f3.value, i, month_loc))
                elif j[0] == 4:
                    values.append(SVM2(Features.f4.value, i, month_loc))
            if j[1] == 'MLR':
                if j[0] == 1:
                    values.append(MLR2(Features.f1.value, i, month_loc))
                elif j[0] == 2:
                    values.append(MLR2(Features.f2.value, i, month_loc))
                elif j[0] == 3:
                    values.append(MLR2(Features.f3.value, i, month_loc))
                elif j[0] == 4:
                    values.append(MLR2(Features.f4.value, i, month_loc))
            if j[1] == 'ANN':
                if j[0] == 1:
                    values.append(ANN2(Features.f1.value, i, month_loc))
                elif j[0] == 2:
                    values.append(ANN2(Features.f2.value, i, month_loc))
                elif j[0] == 3:
                    values.append(ANN2(Features.f3.value, i, month_loc))
                elif j[0] == 4:
                    values.append(ANN2(Features.f4.value, i, month_loc))

    values = pd.DataFrame(values, columns=['Predicted'])
    values['Predicted'] = values['Predicted'].apply(lambda x: negative_checker(x))
    values['Predicted'] = values['Predicted'].apply(lambda x: round(x, 2))
    values['Month'] = range(1, 13)

    values.to_csv(f'outputs/rainfall/{dist},{state}.csv', index=False, header=True)
