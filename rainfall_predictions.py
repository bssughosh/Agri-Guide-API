import pandas as pd
import numpy as np
from sklearn.svm import SVR

pd.options.mode.chained_assignment = None


def rain_caller(state, dist):
    dist = dist.replace('+', ' ')
    state = state.replace('+', ' ')

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = 'rainfall_data.csv'
    df = pd.read_csv(base_url + file)

    selected_state = state
    selected_district = dist

    values = []
    for month_loc, selected_month in enumerate(months):
        df1 = df[df['State'] == selected_state]
        df2 = df1[df1['District'] == selected_district]
        df3 = df2[['Year', selected_month]]
        df3 = pd.DataFrame(df3, columns=['Year', selected_month])

        df3 = df3.sort_values(['Year'])

        x = []
        t = []
        for i, j in df3.iterrows():
            if int(j[0]) != df3.iloc[-1, 0]:
                x.append(j)
            else:
                t.append(j)

        x1 = pd.DataFrame(x, columns=['Year', selected_month])
        x1 = x1.apply(pd.to_numeric, errors='coerce')
        x1 = x1.dropna()
        x1.reset_index(drop=True, inplace=True)

        x2 = pd.DataFrame(t, columns=['Year', selected_month])
        x2 = x2.apply(pd.to_numeric, errors='coerce')
        x2 = x2.dropna()
        x2.reset_index(drop=True, inplace=True)

        forecast_out = 1

        x1['prediction'] = x1[[selected_month]].shift(-forecast_out)

        X = np.array(x1.drop(['Year', 'prediction'], 1))
        X = X[:-forecast_out]

        y = np.array(x1['prediction'])
        y = y[:-forecast_out]

        i = 300
        ind = i
        mini = (np.Inf, np.Inf)

        while i <= 3000:
            svm = SVR(kernel='rbf', C=i)
            svm.fit(X, y)

            x_forecast = np.array(x1.drop(['Year', 'prediction'], 1))[-forecast_out:]
            svm_prediction = svm.predict(x_forecast)
            svm_prediction = pd.DataFrame(svm_prediction, columns=['Predicted'])

            x2.index = svm_prediction.index
            svm_prediction['Original'] = x2[selected_month]

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
        for i, j in df3.iterrows():
            if int(j[0]) <= 2010:
                x.append(j)

        x1 = pd.DataFrame(x, columns=['Year', selected_month])
        x1 = x1.apply(pd.to_numeric, errors='coerce')
        x1 = x1.dropna()
        x1.reset_index(drop=True, inplace=True)

        forecast_out = 2020 - int(x1.iloc[-1, 0])

        x1['prediction'] = x1[[selected_month]].shift(-forecast_out)

        X = np.array(x1.drop(['Year', 'prediction'], 1))
        X = X[:-forecast_out]

        y = np.array(x1['prediction'])
        y = y[:-forecast_out]

        svm = SVR(kernel='rbf', C=ind)
        svm.fit(X, y)

        x_forecast = np.array(x1.drop(['Year', 'prediction'], 1))[-forecast_out:]
        svm_prediction = svm.predict(x_forecast)

        values.append(svm_prediction[-1])

    values = pd.DataFrame(values, columns=['Predicted'])
    values['Month'] = range(1, 13)

    values.to_csv(f'outputs/rainfall/{dist},{state}.csv', index=False, header=True)
