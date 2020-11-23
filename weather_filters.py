import copy

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def fetch_single_loc_rainfall_data(state, dist, years):
    dist1 = dist.replace('+', ' ') if len(dist) > 0 else ''
    state1 = state.replace('+', ' ')

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file1 = 'rainfall_data_3.csv'

    rain = pd.read_csv(base_url + file1)
    rain1 = rain[rain['State'] == state1]
    if dist1 != '':
        rain1 = rain1[rain1['District'] == dist1]

    if len(years) == 2:
        rain1 = rain1[rain1['Year'] >= years[0]]
        rain1 = rain1[rain1['Year'] <= years[1]]

    rain1.reset_index(drop=True, inplace=True)
    return rain1


def fetch_single_loc_temp_data(state, dist, years):
    dist1 = dist.replace('+', ' ') if len(dist) > 0 else ''
    state1 = state.replace('+', ' ')

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file1 = 'whole_temp_2.csv'

    temp = pd.read_csv(base_url + file1)
    temp1 = temp[temp['State'] == state1]
    if dist1 != '':
        temp1 = temp1[temp1['District'] == dist1]

    if len(years) == 2:
        temp1 = temp1[temp1['Year'] >= years[0]]
        temp1 = temp1[temp1['Year'] <= years[1]]

    temp1.reset_index(drop=True, inplace=True)
    return temp1


def fetch_single_loc_humidity_data(state, dist):
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = dist + '%2C' + state + '.csv'
    file = file.replace('+', '%2B')
    df = pd.read_csv(base_url + file)
    cols = ['date_time', 'humidity']
    cols1 = ['humidity']
    df['date_time'] = pd.to_datetime(df['date_time'])
    df1 = df[cols]

    df1['month'] = df1['date_time'].apply(lambda mon: mon.strftime('%m'))
    df1['year'] = df1['date_time'].apply(lambda year: year.strftime('%Y'))

    df1.drop(['date_time'], 1, inplace=True)

    g = df1.groupby(['year', 'month'], as_index=False)

    monthly_averages = g.aggregate(np.mean)
    monthly_averages[cols1] = monthly_averages[cols1].astype('int8')

    df2 = monthly_averages
    cols = ['State', 'District', 'Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
            'Dec']

    res = []
    ys = list(df2['year'].unique())
    ys = ys[:-1]
    for y in ys:
        r = [state.replace('+', ' '), dist.replace('+', ' '), y]
        df3 = df2[df2['year'] == y]
        df3.reset_index(drop=True, inplace=True)
        for i, j in df3.iterrows():
            r.append(j[2])
        res.append(r)
    res = np.array(res)
    res = pd.DataFrame(res, columns=cols)
    return res


def fetch_multiple_states_rainfall_data(states, years):
    states1 = [state.replace('+', ' ') for state in states]

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file1 = 'rainfall_data_3.csv'

    rain = pd.read_csv(base_url + file1)
    first = True

    res = pd.DataFrame(columns=rain.columns)

    for state in states1:
        rain1 = rain[rain['State'] == state]

        if len(years) == 2:
            rain1 = rain1[rain1['Year'] >= years[0]]
            rain1 = rain1[rain1['Year'] <= years[1]]

        rain1.dropna(inplace=True, axis=0)

        if first:
            res = copy.deepcopy(rain1)
            first = False
            res.reset_index(drop=True, inplace=True)
        else:
            res = res.append(rain1, ignore_index=True)
            res.reset_index(drop=True, inplace=True)

    return res


def fetch_multiple_states_temp_data(states, years):
    states1 = [state.replace('+', ' ') for state in states]

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file1 = 'whole_temp_2.csv'

    temp = pd.read_csv(base_url + file1)
    first = True

    res = pd.DataFrame(columns=temp.columns)

    for state in states1:
        temp1 = temp[temp['State'] == state]

        if len(years) == 2:
            temp1 = temp1[temp1['Year'] >= years[0]]
            temp1 = temp1[temp1['Year'] <= years[1]]

        temp1.dropna(inplace=True, axis=0)

        if first:
            res = copy.deepcopy(temp1)
            first = False
            res.reset_index(drop=True, inplace=True)
        else:
            res = res.append(temp1, ignore_index=True)
            res.reset_index(drop=True, inplace=True)

    return res


def fetch_multiple_states_humidity_data(states):
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    places = pd.read_csv(base_url + 'places.csv')
    states_present = places['State'].to_list()
    dists_present = places['District'].to_list()

    res = []
    for state, dist in zip(states_present, dists_present):
        if state in states:
            file = dist + '%2C' + state + '.csv'
            file = file.replace('+', '%2B')
            df = pd.read_csv(base_url + file)
            cols = ['date_time', 'humidity']
            cols1 = ['humidity']
            df['date_time'] = pd.to_datetime(df['date_time'])
            df1 = df[cols]

            df1['month'] = df1['date_time'].apply(lambda mon: mon.strftime('%m'))
            df1['year'] = df1['date_time'].apply(lambda year: year.strftime('%Y'))

            df1.drop(['date_time'], 1, inplace=True)

            g = df1.groupby(['year', 'month'], as_index=False)

            monthly_averages = g.aggregate(np.mean)
            monthly_averages[cols1] = monthly_averages[cols1].astype('int8')

            df2 = monthly_averages

            ys = list(df2['year'].unique())
            ys = ys[:-1]
            for y in ys:
                r = [state.replace('+', ' '), dist.replace('+', ' '), y]
                df3 = df2[df2['year'] == y]
                df3.reset_index(drop=True, inplace=True)
                for i, j in df3.iterrows():
                    r.append(j[2])
                res.append(r)
    res = np.array(res)
    cols = ['State', 'District', 'Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
            'Nov', 'Dec']
    res = pd.DataFrame(res, columns=cols)
    return res


def multiple_states(states, years, params):
    if 'temp' in params:
        temp_res = fetch_multiple_states_temp_data(states, years)
        temp_res.to_csv('filter_outputs/weather/temp.csv')
    if 'humidity' in params:
        humidity_res = fetch_multiple_states_humidity_data(states)
        humidity_res.to_csv('filter_outputs/weather/humidity.csv')
    if 'rainfall' in params:
        rain_res = fetch_multiple_states_rainfall_data(states, years)
        rain_res.to_csv('filter_outputs/weather/rain.csv')
