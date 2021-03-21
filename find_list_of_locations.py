import os

import pandas as pd

# temp_folder = 'outputs/temp'
# humidity_folder = 'outputs/humidity'
# rainfall_folder = 'outputs/rainfall'
#
# temp_files = []
# humidity_files = []
# rain_files = []
#
# for file in os.listdir(temp_folder):
#     if file.split('.')[-1] == 'csv':
#         name = file.split('.')[0]
#         temp_files.append(name)
#
# for file in os.listdir(humidity_folder):
#     if file.split('.')[-1] == 'csv':
#         name = file.split('.')[0]
#         humidity_files.append(name)
#
# for file in os.listdir(rainfall_folder):
#     if file.split('.')[-1] == 'csv':
#         name = file.split('.')[0]
#         rain_files.append(name)
#
# print(len(temp_files))
# print(len(humidity_files))
# print(len(rain_files))
#
# new_states = [i.split(',')[1] for i in rain_files]
# new_dists = [i.split(',')[0] for i in rain_files]
#
# new_places = pd.DataFrame({'State': new_states, 'District': new_dists}, columns=['State', 'District'])
#
# new_places.sort_values(['State', 'District'], inplace=True)
# new_places.reset_index(inplace=True, drop=True)
#
# new_places.to_csv('outputs/datasets/new_places.csv', index=False, header=True)

yield_folder = 'outputs/yield'

yield_files = []

for file in os.listdir(yield_folder):
    if file.split('.')[-1] == 'csv':
        name = file.split('.')[0]
        yield_files.append(name)

# new_states = [i.split(',')[1] for i in rain_files]
# new_dists = [i.split(',')[0] for i in rain_files]

base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/yield/'
file = 'found1_all_18.csv'
df = pd.read_csv(base_url + file)
all_crops = list(df['Crop'].unique())
all_crops_dict = {}
for i, crop in enumerate(all_crops):
    all_crops_dict[str(i)] = crop

# crop_names = list(all_crops_dict.keys())
# crop_ids = [all_crops_dict[i] for i in crop_names]
#
# all_crops_df = pd.DataFrame({'Crop_Name': crop_names, 'Crop_Id': crop_ids}, columns=['Crop_Name', 'Crop_Id'])
#
# all_crops_df.to_csv('outputs/datasets/all_crops.csv', index=False, header=True)

yield_to_write = []
for y in yield_files:
    strings = y.split(',')
    yield_to_write.append(
        [strings[1].replace('+', ' '), strings[0].replace('+', ' '), strings[2], all_crops_dict[strings[3]],
         strings[3]])

formatted_yield = pd.DataFrame(yield_to_write, columns=['State', 'District', 'Season', 'Crop', 'Crop_Id'])

formatted_yield.sort_values(['State', 'District'], inplace=True)
formatted_yield.reset_index(drop=True, inplace=True)

formatted_yield.to_csv('outputs/datasets/found_crop_data.csv', index=False, header=True)
