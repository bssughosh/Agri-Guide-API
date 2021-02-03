import multiprocessing
import os
import time

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

for state, dist in zip(all_states, all_dists):
    try:
        file = dist + ',' + state + '.csv'
        print(f'Started for {dist},{state}')
        tempP = False
        humidP = False
        rainP = False
        if file not in files1:
            tempP = True

        if file not in files2:
            humidP = True

        if file not in files3:
            rainP = True
        p1 = multiprocessing.Process(target=temperature_caller, args=(state, dist,), )
        p2 = multiprocessing.Process(target=humidity_caller, args=(state, dist,), )
        p3 = multiprocessing.Process(target=rain_caller, args=(state, dist,), )

        t0 = time.time()
        if tempP:
            p1.start()
        else:
            print(f'Temperature prediction for {dist},{state} is done')

        t1 = time.time()
        if humidP:
            p2.start()
        else:
            print(f'Humidity prediction for {dist},{state} is done')

        t2 = time.time()
        if rainP:
            p3.start()
        else:
            print(f'Rainfall prediction for {dist},{state} is done')

        if p1.is_alive():
            p1.join()
            print(f'Temperature prediction for {dist},{state} is done in {time.time() - t0}')
        if p2.is_alive():
            p2.join()
            print(f'Humidity prediction for {dist},{state} is done in {time.time() - t1}')
        if p3.is_alive():
            p3.join()
            print(f'Rainfall prediction for {dist},{state} is done in {time.time() - t2}')

    except:
        print(f'Exception occurred for {dist},{state}')
