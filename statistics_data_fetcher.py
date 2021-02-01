import numpy as np
import pandas as pd


def fetch_rainfall_whole_data(state, dist):
    dist1 = dist.replace('+', ' ')
    state1 = state.replace('+', ' ')

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file1 = 'rainfall_data_3.csv'

    rain = pd.read_csv(base_url + file1)
    rain1 = rain[rain['State'] == state1]
    rain1 = rain1[rain1['District'] == dist1]
    rain1.reset_index(drop=True, inplace=True)

    res = []

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    rain1['sum'] = rain1[months].sum(axis=1)

    for i, j in rain1.iterrows():
        res.append({str(int(j[2])): round(j[15], 2)})

    return res


def fetch_temp_whole_data(state, dist):
    dist1 = dist.replace('+', ' ')
    state1 = state.replace('+', ' ')

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file2 = 'whole_temp_2.csv'

    temp = pd.read_csv(base_url + file2)
    temp1 = temp[temp['State'] == state1]
    temp1 = temp1[temp1['District'] == dist1]
    temp1.reset_index(drop=True, inplace=True)

    res = []

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    temp1['sum'] = temp1[months].mean(axis=1)

    for i, j in temp1.iterrows():
        res.append({str(int(j[2])): round(j[15], 2)})

    return res


def fetch_humidity_whole_data(state, dist):
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = dist + '%2C' + state + '.csv'
    file = file.replace('+', '%2B')
    print(base_url + file)
    humidity = pd.read_csv(base_url + file)

    h1 = humidity[['date_time', 'humidity']]
    h1['date_time'] = pd.to_datetime(h1['date_time'])

    h1['month'] = h1['date_time'].apply(lambda mon: mon.strftime('%m'))
    h1['year'] = h1['date_time'].apply(lambda year: year.strftime('%Y'))

    h1.drop(['date_time'], 1, inplace=True)

    g1 = h1.groupby(['year'], as_index=False)
    yearly_averages = g1.aggregate(np.mean)
    yearly_averages[['humidity']] = yearly_averages[['humidity']].astype('int8')
    h3 = yearly_averages

    res = []

    for i, j in h3.iterrows():
        res.append({str(int(j[0])): round(j[1], 0)})

    return res
