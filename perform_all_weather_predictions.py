import os
import time
from csv import writer

import pandas as pd

from humidity_predictions import humidity_caller
from rainfall_predictions import rain_caller
from temp_predictions import temperature_caller

base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
places_file = 'places.csv'

places = pd.read_csv(base_url + places_file)

all_states = list(places['State'])
all_dists = list(places['District'])

files1 = os.listdir('outputs/temp')
files2 = os.listdir('outputs/humidity')
files3 = os.listdir('outputs/rainfall')

for state, dist in zip(all_states, all_dists):
    exception_file = pd.read_csv('outputs/exceptions.csv')
    try:
        file = dist + ',' + state + '.csv'
        print(f'Started for {dist},{state}')
        t0 = time.time()
        if file not in files1:
            temperature_caller(state, dist)
        print(f'Temperature prediction for {dist},{state} is done in {time.time() - t0}')

        t1 = time.time()
        if file not in files2:
            humidity_caller(state, dist)
        print(f'Humidity prediction for {dist},{state} is done in {time.time() - t1}')

        t2 = time.time()
        if file not in files3:
            exception_file_1 = exception_file[exception_file['State'] == state]
            exception_file_1 = exception_file_1[exception_file_1['District'] == dist]
            if exception_file_1.shape[0] == 0:
                rain_caller(state, dist)
        print(f'Rainfall prediction for {dist},{state} is done in {time.time() - t2}')

    except:
        print(f'Exception occurred for {dist},{state}')
        with open('outputs/exceptions.csv', 'a+', newline='') as fd:
            csv_writer = writer(fd)
            csv_writer.writerow([state, dist])
