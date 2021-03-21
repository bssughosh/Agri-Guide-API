import os

import pandas as pd

temp_folder = 'outputs/temp'
humidity_folder = 'outputs/humidity'
rainfall_folder = 'outputs/rainfall'

temp_files = []
humidity_files = []
rain_files = []

for file in os.listdir(temp_folder):
    if file.split('.')[-1] == 'csv':
        name = file.split('.')[0]
        temp_files.append(name)

for file in os.listdir(humidity_folder):
    if file.split('.')[-1] == 'csv':
        name = file.split('.')[0]
        humidity_files.append(name)

for file in os.listdir(rainfall_folder):
    if file.split('.')[-1] == 'csv':
        name = file.split('.')[0]
        rain_files.append(name)

print(len(temp_files))
print(len(humidity_files))
print(len(rain_files))

new_states = [i.split(',')[1] for i in rain_files]
new_dists = [i.split(',')[0] for i in rain_files]

new_places = pd.DataFrame({'State': new_states, 'District': new_dists}, columns=['State', 'District'])

new_places.sort_values(['State', 'District'], inplace=True)
new_places.reset_index(inplace=True, drop=True)

new_places.to_csv('outputs/datasets/new_places.csv', index=False, header=True)
