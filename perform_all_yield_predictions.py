import os
import time
from collections import Counter
from csv import writer

import pandas as pd

from yield_prediction import yield_caller

base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/yield/'
data_file = 'found1_all_18.csv'

data = pd.read_csv(base_url + data_file)

all_states = list(data['State'])
all_dists = list(data['District'])
all_seasons = list(data['Season'])
all_crops = list(data['Crop'])

files = os.listdir('outputs/yield')

total_count = 0
states_total = Counter(all_states)
states_count = 0
previous_state = all_states[0]

crops = list(data['Crop'].unique())

for state, dist, season, crop in zip(all_states, all_dists, all_seasons, all_crops):
    crop_id = crops.index(crop)
    dist = dist.replace(' ', '+')
    state = state.replace(' ', '+')
    exception_file = pd.read_csv('outputs/exceptions_yield.csv')
    total_count = total_count + 1
    if previous_state == state:
        states_count = states_count + 1
    else:
        previous_state = state
        states_count = 1
    try:
        file = dist + ',' + state + ',' + season + ',' + str(crop_id) + '.csv'
        print(f'Overall Count => {total_count}/{len(all_dists)}')
        print(f'State Count => {states_count}/{states_total[state.replace("+", " ")]}')
        print(f'Started for {dist},{state} => {season},{crop}')
        exception_file1 = exception_file[exception_file['State'] == state]
        exception_file1 = exception_file1[exception_file1['District'] == dist]
        exception_file1 = exception_file1[exception_file1['Season'] == season]
        exception_file1 = exception_file1[exception_file1['Crop'] == crop]
        t0 = time.time()
        if file not in files:
            if exception_file1.shape[0] == 0:
                yield_caller(state, dist, season, str(crop_id))
        print(f'Yield prediction for {dist},{state} => {season},{crop} is done in {time.time() - t0}')

    except:
        print(f'Exception occurred for {dist},{state} => {season},{crop}')
        with open('outputs/exceptions_yield.csv', 'a+', newline='') as fd:
            csv_writer = writer(fd)
            csv_writer.writerow([state, dist, season, crop])
