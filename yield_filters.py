import copy

import pandas as pd


def fetch_single_loc_yield_data(state, dist):
    dist1 = dist.replace('+', ' ') if len(dist) > 0 else ''
    state1 = state.replace('+', ' ')

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/yield/'
    file1 = 'Master_data_1.csv'

    yield_data = pd.read_csv(base_url + file1)
    yield_data1 = yield_data[yield_data['State_Name'] == state1]
    if dist1 != '':
        yield_data1 = yield_data1[yield_data1['District_Name'] == dist1]

    yield_data1.reset_index(drop=True, inplace=True)
    return yield_data1


def single_loc_yield(state, dist):
    yield_res = fetch_single_loc_yield_data(state, dist)
    yield_res.to_csv('filter_outputs/yield/yield.csv')


def fetch_multiple_dists_yield_data(state, dists):
    state1 = state.replace('+', ' ')
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/yield/'
    file1 = 'Master_data_1.csv'

    if len(dists) == 1 and '0' in dists:
        places = pd.read_csv(base_url + 'found1_all_18.csv')
        places = places[places['State'] == state1]
        final_dists = places['District'].to_list()
        final_dists = [dist.replace('+', ' ') for dist in final_dists]
    else:
        final_dists = [dist.replace('+', ' ') for dist in dists]

    yield_data = pd.read_csv(base_url + file1)
    yield_data = yield_data[yield_data['State_Name'] == state1]
    first = True

    res = pd.DataFrame(columns=yield_data.columns)

    for dist in final_dists:
        yield_data1 = yield_data[yield_data['District_Name'] == dist]

        yield_data1.dropna(inplace=True, axis=0)

        if first:
            res = copy.deepcopy(yield_data1)
            first = False
            res.reset_index(drop=True, inplace=True)
        else:
            res = res.append(yield_data1, ignore_index=True)
            res.reset_index(drop=True, inplace=True)

    return res


def multiple_dists_yield(state, dists):
    yield_res = fetch_multiple_dists_yield_data(state, dists)
    yield_res.to_csv('filter_outputs/yield/yield.csv')


def fetch_multiple_states_yield_data(states):
    states1 = [state.replace('+', ' ') for state in states]

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/yield/'
    file1 = 'Master_data_1.csv'

    yield_data = pd.read_csv(base_url + file1)
    first = True

    res = pd.DataFrame(columns=yield_data.columns)

    for state in states1:
        yield_data1 = yield_data[yield_data['State_Name'] == state]

        yield_data1.dropna(inplace=True, axis=0)

        if first:
            res = copy.deepcopy(yield_data1)
            first = False
            res.reset_index(drop=True, inplace=True)
        else:
            res = res.append(yield_data1, ignore_index=True)
            res.reset_index(drop=True, inplace=True)

    return res


def multiple_states_yield(states):
    yield_res = fetch_multiple_states_yield_data(states)
    yield_res.to_csv('filter_outputs/yield/yield.csv')
