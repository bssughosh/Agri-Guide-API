import os
import time
from collections import Counter
from csv import writer

import pandas as pd

from humidity_predictions import humidity_caller
from rainfall_predictions_new import rain_caller
from temp_predictions import temperature_caller

base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
places_file = 'places.csv'

places = pd.read_csv(base_url + places_file)

all_states = list(places['State'])
all_dists = list(places['District'])

files1 = os.listdir('outputs/temp')
files2 = os.listdir('outputs/humidity')
files3 = os.listdir('outputs/rainfall')

total_count = 0
states_total = Counter(all_states)
states_count = 0
previous_state = all_states[0]

for state, dist in zip(all_states, all_dists):
    exception_file = pd.read_csv('outputs/exceptions.csv')
    total_count = total_count + 1
    if previous_state == state:
        states_count = states_count + 1
    else:
        previous_state = state
        states_count = 1
    if state == 'gujarat':
        continue
    try:
        file = dist + ',' + state + '.csv'
        print(f'Overall Count => {total_count}/{len(all_dists)}')
        print(f'State Count => {states_count}/{states_total[state]}')
        print(f'Started for {dist},{state}')
        t0 = time.time()
        if file not in files1:
            exception_file_1 = exception_file[exception_file['State'] == state]
            exception_file_1 = exception_file_1[exception_file_1['District'] == dist]
            if exception_file_1.shape[0] == 0:
                temperature_caller(state, dist)
        print(f'Temperature prediction for {dist},{state} is done in {time.time() - t0}')

        t1 = time.time()
        if file not in files2:
            exception_file_1 = exception_file[exception_file['State'] == state]
            exception_file_1 = exception_file_1[exception_file_1['District'] == dist]
            if exception_file_1.shape[0] == 0:
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
