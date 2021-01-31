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
