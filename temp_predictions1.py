import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

pd.options.mode.chained_assignment = None


def temperature_caller(state, dist):
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = dist + '%2C' + state + '.csv'
    file = file.replace('+', '%2B')
    df = pd.read_csv(base_url + file)
    cols = ['date_time', 'maxtempC', 'mintempC', 'cloudcover', 'visibility', 'windspeedKmph', 'FeelsLikeC', 'tempC']
    cols1 = ['maxtempC', 'mintempC', 'cloudcover', 'visibility', 'windspeedKmph', 'FeelsLikeC', 'tempC']
    cols2 = ['year', 'month', 'maxtempC', 'mintempC', 'cloudcover', 'visibility', 'windspeedKmph', 'FeelsLikeC',
             'tempC']
    df['date_time'] = pd.to_datetime(df['date_time'])
    df1 = df[cols]

    df1['month'] = df1['date_time'].apply(lambda mon: mon.strftime('%m'))
    df1['year'] = df1['date_time'].apply(lambda year: year.strftime('%Y'))

    df1.drop(['date_time'], 1, inplace=True)

    g = df1.groupby(['year', 'month'], as_index=False)

    monthly_averages = g.aggregate(np.mean)
    monthly_averages[cols1] = monthly_averages[cols1].astype('int8')

    df2 = monthly_averages
    ####################################################################################

    x = []
    t = []
    for i, j in df2.iterrows():
        if int(j[0]) < 2018:
            x.append(j)
        elif int(j[0]) == 2018:
            t.append(j)

    x1 = pd.DataFrame(x, columns=cols2)
    x2 = pd.DataFrame(t, columns=cols2)

    forecast_out = 12

    x1['prediction'] = x1[['tempC']].shift(-forecast_out)

    X = np.array(x1.drop(['prediction', 'year', 'month'], 1))
    X = X[:-forecast_out]

    y = np.array(x1['prediction'])
    y = y[:-forecast_out]

    t0 = time.time()
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': list(range(1, 3000, 49)),
        },
        scoring='neg_mean_squared_error',
        n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    print(best_params['C'])
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': list(range(best_params['C'], best_params['C'] + 50)),
        },
        scoring='neg_mean_squared_error',
        n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    print(best_params['C'])
    ind = best_params['C']

    print(time.time() - t0)

    x = []
    og = []

    for i, j in df2.iterrows():
        if int(j[0]) <= 2018:
            x.append(j)
        if int(j[0]) == 2019:
            og.append(int(j[8]))

    x1 = pd.DataFrame(x, columns=cols2)
    forecast_out = 12
    x1['prediction'] = x1[['tempC']].shift(-forecast_out)

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
    svm_prediction['Original'] = og
    svm_prediction['Error'] = abs(svm_prediction['Original'] - svm_prediction['Predicted'])
    svm_prediction['Accuracy'] = ((svm_prediction['Original'] - svm_prediction['Error']) * 100) / (svm_prediction[
        'Original'])
    svm_prediction['Month'] = range(1, 13)

    print(svm_prediction)
    svm_prediction.to_csv(f'temp/{dist},{state}.csv', index=False)


temperature_caller('tamil+nadu', 'coimbatore')
