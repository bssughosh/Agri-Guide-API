import enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

pd.options.mode.chained_assignment = None


class Features(enum.Enum):
    f1 = ['Rain', 'Temp']
    f2 = ['Rain']
    f3 = ['Temp']


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


def SVM1(data, feature_set):
    independent_variables = feature_set
    dependent = 'Pointer'

    X = np.array(data[independent_variables][:-2])
    y = np.array(data[dependent][:-2])
    x_forecast = np.array(data[independent_variables][-2:-1])
    y2 = np.array(data[dependent][-2:-1])

    i = 300
    ind = i
    mini = (np.Inf, np.Inf)

    while i <= 3000:
        svm = SVR(kernel='rbf', C=i)
        svm.fit(X, y)

        svm_prediction = svm.predict(x_forecast)
        svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
        svm_prediction['Original'] = y2

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

    X = np.array(data[independent_variables][:-1])
    y = np.array(data[dependent][:-1])
    x_forecast = np.array(data[independent_variables][-1:])

    svm = SVR(kernel='rbf', C=ind)
    svm.fit(X, y)

    svm_prediction = svm.predict(x_forecast)
    return svm_prediction[-1]


def MLR1(data, feature_set):
    independent_variables = feature_set
    dependent = 'Pointer'

    X = np.array(data[independent_variables][:-1])
    y = np.array(data[dependent][:-1])
    x_forecast = np.array(data[independent_variables][-1:])

    lr = LinearRegression()
    lr.fit(X, y)

    lr_forecast = lr.predict(x_forecast)
    return lr_forecast[-1]


def ANN1(data, selected_season, feature_set):
    independent_variables = feature_set
    dependent = ['Pointer']

    X = np.array(data[independent_variables][:-1])
    y = np.array(data[dependent][:-1])
    x_forecast = np.array(data[independent_variables][-1:])

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    model = LinearRegressionModel(input_dim, output_dim)

    criterion = nn.MSELoss()
    if selected_season == 'Kharif':
        learning_rate = 0.0000001
    else:
        learning_rate = 0.000001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 200

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


def SVM2(data, feature_set, state, dist):
    independent_variables = feature_set
    dependent = 'Pointer'

    X = np.array(data[independent_variables][:-1])
    y = np.array(data[dependent][:-1])
    x_forecast = np.array(data[independent_variables][-1:])
    y2 = np.array(data[dependent][-1:])

    i = 300
    ind = i
    mini = (np.Inf, np.Inf)

    while i <= 3000:
        svm = SVR(kernel='rbf', C=i)
        svm.fit(X, y)

        svm_prediction = svm.predict(x_forecast)
        svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])
        svm_prediction['Original'] = y2

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

    X = np.array(data[independent_variables])
    y = np.array(data[dependent])

    temp_2020_df = pd.read_csv(f'outputs/temp/{dist},{state}.csv')
    temp_2019 = temp_2020_df['Predicted'].to_list()
    del temp_2020_df

    rain_2020_df = pd.read_csv(f'outputs/rainfall/{dist},{state}.csv')
    rain_2019 = rain_2020_df['Predicted'].to_list()
    del rain_2020_df

    if independent_variables == Features.f1.value:
        x_forecast = np.array([sum(rain_2019), (sum(temp_2019) / len(temp_2019))])
    elif independent_variables == Features.f2.value:
        x_forecast = np.array([sum(rain_2019), ]).reshape(-1, 1)
    else:
        x_forecast = np.array([(sum(temp_2019) / len(temp_2019)), ]).reshape(-1, 1)

    svm = SVR(kernel='rbf', C=ind)
    svm.fit(X, y)

    svm_prediction = svm.predict(x_forecast)
    return svm_prediction[-1]


def MLR2(data, feature_set, state, dist):
    independent_variables = feature_set
    dependent = 'Pointer'

    X = np.array(data[independent_variables])
    y = np.array(data[dependent])
    temp_2020_df = pd.read_csv(f'outputs/temp/{dist},{state}.csv')
    temp_2019 = temp_2020_df['Predicted'].to_list()
    del temp_2020_df

    rain_2020_df = pd.read_csv(f'outputs/rainfall/{dist},{state}.csv')
    rain_2019 = rain_2020_df['Predicted'].to_list()
    del rain_2020_df

    if independent_variables == Features.f1.value:
        x_forecast = np.array([sum(rain_2019), (sum(temp_2019) / len(temp_2019))])
    elif independent_variables == Features.f2.value:
        x_forecast = np.array([sum(rain_2019), ]).reshape(-1, 1)
    else:
        x_forecast = np.array([(sum(temp_2019) / len(temp_2019)), ]).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(X, y)

    lr_forecast = lr.predict(x_forecast)
    return lr_forecast[-1]


def ANN2(data, selected_season, feature_set, state, dist):
    independent_variables = feature_set
    dependent = ['Pointer']

    X = np.array(data[independent_variables])
    y = np.array(data[dependent])

    temp_2020_df = pd.read_csv(f'outputs/temp/{dist},{state}.csv')
    temp_2019 = temp_2020_df['Predicted'].to_list()
    del temp_2020_df

    rain_2020_df = pd.read_csv(f'outputs/rainfall/{dist},{state}.csv')
    rain_2019 = rain_2020_df['Predicted'].to_list()
    del rain_2020_df

    if independent_variables == Features.f1.value:
        x_forecast = np.array([sum(rain_2019), (sum(temp_2019) / len(temp_2019))])
    elif independent_variables == Features.f2.value:
        x_forecast = np.array([sum(rain_2019), ]).reshape(-1, 1)
    else:
        x_forecast = np.array([(sum(temp_2019) / len(temp_2019)), ]).reshape(-1, 1)

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    model = LinearRegressionModel(input_dim, output_dim)

    criterion = nn.MSELoss()
    if selected_season == 'Kharif':
        learning_rate = 0.0000001
    else:
        learning_rate = 0.000001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 200

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


def yield_caller(state, dist, season, crop_id):
    dist1 = dist.replace('+', ' ')
    state1 = state.replace('+', ' ')
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/'
    file = 'yield/found1_all_18.csv'

    df = pd.read_csv(base_url + file)
    all_crops = list(df['Crop'].unique())
    all_crops_dict = {}
    for i, crop in enumerate(all_crops):
        all_crops_dict[str(i)] = crop

    file1 = 'yield/Master_data_1.csv'

    mas = pd.read_csv(base_url + file1)

    mas1 = mas[mas['State_Name'] == state1]
    mas1 = mas1[mas1['District_Name'] == dist1]

    mas1.reset_index(drop=True, inplace=True)
    cols = ['Season', 'Crop']
    for col in cols:
        mas1[col] = mas1[col].apply(lambda x: x.strip())

    mas2 = mas1[mas1['Season'] == season]
    mas2 = mas2[mas2['Crop'] == all_crops_dict[crop_id]]

    mas2.reset_index(drop=True, inplace=True)

    mas2['Pointer'] = (mas2['Production'] * 10) / mas2['Area']

    if season == 'Kharif':
        rain_cols = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        temp_cols = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    else:
        rain_cols = ['Year', 'Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        temp_cols = ['Year', 'Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']

    file2 = 'weather/rainfall_data_3.csv'
    rain = pd.read_csv(base_url + file2)
    rain1 = rain[rain['State'] == state1]
    rain1 = rain1[rain1['District'] == dist1]
    rain1.reset_index(drop=True, inplace=True)
    rain2 = []
    for i, j in rain1.iterrows():
        if 1997 <= int(j[2]) <= 2014:
            rain2.append(j)
    rain2 = pd.DataFrame(rain2, columns=rain1.columns)
    rain2.reset_index(drop=True, inplace=True)
    rain3 = rain2[rain_cols]

    file3 = 'weather/whole_temp_2.csv'
    temp = pd.read_csv(base_url + file3)
    temp1 = temp[temp['State'] == state1]
    temp1 = temp1[temp1['District'] == dist1]
    temp1.reset_index(drop=True, inplace=True)
    temp2 = []
    for i, j in temp1.iterrows():
        if 1997 <= int(j[2]) <= 2014:
            temp2.append(j)
    temp2 = pd.DataFrame(temp2, columns=temp1.columns)
    temp2.reset_index(drop=True, inplace=True)
    temp3 = temp2[temp_cols]

    temp_cols = temp_cols[1:]
    rain_cols = rain_cols[1:]

    temp_avg = temp3[temp_cols].mean(axis=1)
    rain_sum = rain3[rain_cols].sum(axis=1)

    compiled_data = pd.DataFrame()
    compiled_data['Production'] = mas2['Production']
    compiled_data['Area'] = 10
    compiled_data['Pointer'] = mas2['Pointer']
    compiled_data['Temp'] = temp_avg
    compiled_data['Rain'] = rain_sum

    data_after_outlier_analysis = compiled_data['Pointer'].between(compiled_data['Pointer'].quantile(.05),
                                                                   compiled_data['Pointer'].quantile(.95), )
    indices_to_drop = []
    for i, j in enumerate(data_after_outlier_analysis):
        if not j:
            indices_to_drop.append(i)
    compiled_data = compiled_data.drop(indices_to_drop)
    compiled_data.reset_index(drop=True, inplace=True)

    values = [compiled_data.iloc[-1, 2]]  # Actual Value
    for feature in Features:
        values.append(SVM1(compiled_data, feature.value))
        values.append(MLR1(compiled_data, feature.value))
        ann_result = ANN1(compiled_data, season, feature.value)
        while ann_result < 0:
            ann_result = ANN1(compiled_data, season, feature.value)
        values.append(ann_result)

    diff = [abs(values[0] - i) for i in values]
    diff = diff[1:]

    best_combination_index = diff.index(min(diff))
    res = 0
    if best_combination_index == 0:
        res = SVM2(compiled_data, Features.f1.value, state, dist)
    elif best_combination_index == 1:
        res = MLR2(compiled_data, Features.f1.value, state, dist)
    elif best_combination_index == 2:
        res = ANN2(compiled_data, season, Features.f1.value, state, dist)
        while res < 0:
            res = ANN2(compiled_data, season, Features.f1.value, state, dist)

    elif best_combination_index == 3:
        res = SVM2(compiled_data, Features.f2.value, state, dist)
    elif best_combination_index == 4:
        res = MLR2(compiled_data, Features.f2.value, state, dist)
    elif best_combination_index == 5:
        res = ANN2(compiled_data, season, Features.f2.value, state, dist)
        while res < 0:
            res = ANN2(compiled_data, season, Features.f2.value, state, dist)

    elif best_combination_index == 6:
        res = SVM2(compiled_data, Features.f3.value, state, dist)
    elif best_combination_index == 7:
        res = MLR2(compiled_data, Features.f3.value, state, dist)
    elif best_combination_index == 8:
        res = ANN2(compiled_data, season, Features.f3.value, state, dist)
        while res < 0:
            res = ANN2(compiled_data, season, Features.f3.value, state, dist)

    values = np.array([round(res, 3), ]).reshape(-1, 1)
    values = pd.DataFrame(values, columns=['Predicted'])

    values.to_csv(f'outputs/yield/{dist},{state},{season},{crop_id}.csv', index=False, header=True)
