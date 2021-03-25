import os
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from flask import Flask, jsonify, send_from_directory, request, send_file
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from statistics_data_fetcher import fetch_rainfall_whole_data, fetch_temp_whole_data, fetch_humidity_whole_data
from weather_filters import multiple_states, single_loc, multiple_dists
from yield_filters import multiple_states_yield, single_loc_yield, multiple_dists_yield

app = Flask(__name__)
auth = HTTPBasicAuth()
CORS(app)

_keyNameTemperature = 'temperature'
_keyNameHumidity = 'humidity'
_keyNameRainfall = 'rainfall'
_keyNameDists = 'dists'
_keyNameDistrict = 'district'
_keyNameState = 'state'
_keyNameStates = 'states'
_keyNameId = 'id'
_keyNameStateId = 'state_id'
_keyNameSeasons = 'seasons'
_keyNameCrops = 'crops'
_keyNameYield = 'yield'
_keyNameName = 'name'
_keyNameCropId = 'crop_id'

_queryParamState = 'state'
_queryParamDist = 'dist'
_queryParamStates = 'states'
_queryParamDists = 'dists'
_queryParamYears = 'years'
_queryParamParams = 'params'
_queryParamStateId = 'state_id'
_queryParamDistId = 'dist_id'
_queryParamSeason = 'season'
_queryParamCrop = 'crop'

users = {
    "sughosh": generate_password_hash("hello")
}


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        print('Authenticated')
        return username


@app.route('/')
def home():
    print(f'/home endpoint called ')
    return 'Agri Guide'


@app.route('/weather')
def weather():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    state = state.replace(' ', '+')
    dist = dist.replace(' ', '+')
    print(f'/weather endpoint called with state={state} and dist={dist}')
    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    if state == 'Test' and dist == 'Test':
        return jsonify({_keyNameTemperature: [1, ],
                        _keyNameHumidity: [2, ],
                        _keyNameRainfall: [3, ]})

    files1 = os.listdir('outputs/temp')
    files2 = os.listdir('outputs/humidity')
    files3 = os.listdir('outputs/rainfall')

    file = dist + ',' + state + '.csv'
    try:
        if file not in files1:
            # temperature_caller(state, dist)
            return jsonify({'message': 'The requested location cannot be processed'}), 404

        if file not in files2:
            # humidity_caller(state, dist)
            return jsonify({'message': 'The requested location cannot be processed'}), 404

        if file not in files3:
            # rain_caller(state, dist)
            return jsonify({'message': 'The requested location cannot be processed'}), 404

        print(f'All weather prediction complete for state={state} and dist={dist}')

        df1 = pd.read_csv(f'outputs/temp/{file}')
        df2 = pd.read_csv(f'outputs/humidity/{file}')
        df3 = pd.read_csv(f'outputs/rainfall/{file}')

        my_values = {
            _keyNameTemperature: df1['Predicted'].to_list(),
            _keyNameHumidity: df2['Predicted'].to_list(),
            _keyNameRainfall: df3['Predicted'].round(2).to_list()
        }

        return jsonify(my_values), 200

    except FileNotFoundError:
        return jsonify({'message': 'The requested location cannot be processed'}), 404


def create_file_name():
    present_time = datetime.now()
    _day = present_time.day
    _month = present_time.month
    _year = present_time.year
    _minute = present_time.minute
    _hour = present_time.hour
    _second = present_time.second

    _filename = str(_year) + '_' + str(_month) + '_' + str(_day) + '_' + str(_hour) + '_' + str(_minute) + '_' + str(
        _second) + '.zip'

    return _filename


def clear_file_contents(param):
    if param == 'yield':
        file = open(f'filter_outputs/yield/{param}.csv', 'w+')
        file.close()
    else:
        file = open(f'filter_outputs/weather/{param}.csv', 'w+')
        file.close()


@app.route('/agri_guide/downloads')
def download_with_filters():
    """
    states: List of states\n
    dists: Will be used only when len(states) == 1\n
    years: If years == 0 then all years else will accept length of 2\n
    params: temp,humidity,rainfall,yield\n
    :return: ZIP file containing the required CSV files
    """
    states = request.args.getlist(_queryParamStates)
    dists = request.args.getlist(_queryParamDists)
    years = request.args.getlist(_queryParamYears)
    params = request.args.getlist(_queryParamParams)
    try:
        if len(states) == 1:
            states = states[0].split(',')
            states = [state.replace(' ', '+') for state in states]
        if len(dists) == 1:
            dists = dists[0].split(',')
            dists = [dist.replace(' ', '+') for dist in dists]
        if len(years) == 1:
            years = years[0].split(',')
            years = [int(i) for i in years]
        if len(params) == 1:
            params = params[0].split(',')

        print(f'/agri_guide/downloads endpoint called with states={states}, '
              f'dists={dists}, years={years} and params={params}')

        clear_file_contents('temp')
        clear_file_contents('humidity')
        clear_file_contents('rain')
        clear_file_contents('yield')

        if len(states) > 1:
            multiple_states(states, years, params)

        if len(states) == 1 and len(dists) > 1:
            multiple_dists(states[0], dists, years, params)

        if len(states) == 1 and len(dists) == 1:
            if dists == ['0']:
                multiple_dists(states[0], dists, years, params)
            else:
                single_loc(states[0], dists[0], years, params)
        try:
            if 'yield' in params:
                if len(states) > 1:
                    multiple_states_yield(states)

                if len(states) == 1 and len(dists) > 1:
                    multiple_dists_yield(states[0], dists)

                if len(states) == 1 and len(dists) == 1:
                    if dists == ['0']:
                        multiple_dists_yield(states[0], dists)
                    else:
                        single_loc_yield(states[0], dists[0])
        except:
            print(f'yield data not found for states={states} and dists={dists}')

        handle = ZipFile('required_downloads.zip', 'w')
        if 'temp' in params:
            handle.write('filter_outputs/weather/temp.csv', 'temperature.csv', compress_type=ZIP_DEFLATED)
        if 'humidity' in params:
            handle.write('filter_outputs/weather/humidity.csv', 'humidity.csv', compress_type=ZIP_DEFLATED)
        if 'rainfall' in params:
            handle.write('filter_outputs/weather/rain.csv', 'rainfall.csv', compress_type=ZIP_DEFLATED)
        if 'yield' in params:
            handle.write('filter_outputs/yield/yield.csv', 'yield.csv', compress_type=ZIP_DEFLATED)
        handle.close()

        print(f'ZipFile created for states={states}, '
              f'dists={dists}, years={years} and params={params}')

        response = send_file('required_downloads.zip', as_attachment=True, attachment_filename=create_file_name())

        return response, 200

    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404


@app.route('/weather/downloads')
def download_weather_filters():
    """
    states: List of states\n
    dists: Will be used only when len(states) == 1\n
    years: If years == 0 then all years else will accept length of 2\n
    params: temp,humidity,rainfall\n
    :return: ZIP file containing the required CSV files
    """
    states = request.args.getlist(_queryParamStates)
    dists = request.args.getlist(_queryParamDists)
    years = request.args.getlist(_queryParamYears)
    params = request.args.getlist(_queryParamParams)
    try:
        if len(states) == 1:
            states = states[0].split(',')
            states = [state.replace(' ', '+') for state in states]
        if len(dists) == 1:
            dists = dists[0].split(',')
            dists = [dist.replace(' ', '+') for dist in dists]
        if len(years) == 1:
            years = years[0].split(',')
            years = [int(i) for i in years]
        if len(params) == 1:
            params = params[0].split(',')

        print(f'/weather/downloads endpoint called with states={states}, '
              f'dists={dists}, years={years} and params={params}')

        if len(states) > 1:
            multiple_states(states, years, params)

        if len(states) == 1 and len(dists) > 1:
            multiple_dists(states[0], dists, years, params)

        if len(states) == 1 and len(dists) == 1:
            if dists == ['0']:
                multiple_dists(states[0], dists, years, params)
            else:
                single_loc(states[0], dists[0], years, params)

        handle = ZipFile('required_downloads.zip', 'w')
        if 'temp' in params:
            handle.write('filter_outputs/weather/temp.csv', 'temperature.csv', compress_type=ZIP_DEFLATED)
        if 'humidity' in params:
            handle.write('filter_outputs/weather/humidity.csv', 'humidity.csv', compress_type=ZIP_DEFLATED)
        if 'rainfall' in params:
            handle.write('filter_outputs/weather/rain.csv', 'rainfall.csv', compress_type=ZIP_DEFLATED)

        handle.close()

        print(f'ZipFile created for states={states}, '
              f'dists={dists}, years={years} and params={params}')

        response = send_file('required_downloads.zip', as_attachment=True, attachment_filename=create_file_name())

        return response, 200

    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404


@app.route('/weather/files')
@auth.login_required
def download_weather_predicted_files():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    state = state.replace(' ', '+')
    dist = dist.replace(' ', '+')

    print(f'/weather/files endpoint called with state={state} and '
          f'dist={dist}')
    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    files3 = os.listdir('outputs/rainfall')

    file = dist + ',' + state + '.csv'
    if file in files3:
        handle = ZipFile(f'{dist},{state}.zip', 'w')
        handle.write(f'outputs/temp/{file}', 'temperature.csv', compress_type=ZIP_DEFLATED)
        handle.write(f'outputs/humidity/{file}', 'humidity.csv', compress_type=ZIP_DEFLATED)
        handle.write(f'outputs/temp/{file}', 'rainfall.csv', compress_type=ZIP_DEFLATED)
        handle.close()

        print(f'ZipFile created for state={state} and '
              f'dist={dist}')

        return send_from_directory('', f'{dist},{state}.zip', as_attachment=True)
    else:
        return jsonify({'message': 'File not found'}), 404


def preprocessing(s):
    s = s.replace('+', ' ')
    s = s.title()
    return s


@app.route('/get_states')
def get_state():
    isTest = request.args.get('isTest')
    print(f'/get_states endpoint called')
    base_url = 'outputs/datasets/'
    file = 'new_places.csv'
    df = pd.read_csv(base_url + file)

    df['State'] = df['State'].apply(lambda c: preprocessing(c))
    res = {}
    res1 = []

    states = list(df['State'].unique())

    for i, j in enumerate(states):
        t = {_keyNameId: str(i + 1), _keyNameName: j}
        res1.append(t)

    if isTest == 'true':
        return jsonify({_keyNameState: [{_keyNameId: 'Test', _keyNameName: 'Test'}, ]})

    res[_keyNameState] = res1
    return jsonify(res), 200


@app.route('/get_state_value')
def get_state_for_state_id():
    state_id = request.args.getlist(_queryParamStateId)
    base_url = 'outputs/datasets/'
    file = 'new_places.csv'
    df = pd.read_csv(base_url + file)
    if len(state_id) == 1:
        state_id = state_id[0].split(',')
        state_id = [(int(s) - 1) for s in state_id]
    print(f'/get_state_value endpoint called with state_id={state_id}')
    if state_id == [1000, ]:
        return jsonify({_keyNameStates: ['Test', ]})
    states = list(df['State'].unique())
    res = []
    for s in state_id:
        res.append(states[s])

    print(f'/get_state_value endpoint returned => {res}')

    return jsonify({_keyNameStates: res}), 200


@app.route('/get_dists')
def get_dist():
    state_id = request.args.get(_queryParamStateId)
    if state_id is None:
        return jsonify({'message': 'State ID not found'}), 404
    try:
        state_id = int(state_id)
        if state_id == 1000:
            return jsonify({_keyNameDistrict: [{_keyNameId: 'Test', _keyNameStateId: 'Test', _keyNameName: 'Test'}, ]})
    except ValueError:
        return jsonify({'message': 'State ID not found'}), 404

    print(f'/get_dists endpoint called with state_id={state_id}')

    base_url = 'outputs/datasets/'
    file = 'new_places.csv'
    df = pd.read_csv(base_url + file)
    df['State'] = df['State'].apply(lambda c: preprocessing(c))
    df['District'] = df['District'].apply(lambda c: preprocessing(c))
    res = {}
    res1 = []

    k = 1
    p = df.iloc[0, 0]
    for i, j in df.iterrows():
        if j[0] != p:
            k += 1
            p = j[0]
        if state_id == k:
            t = {_keyNameId: str(i), _keyNameStateId: str(k), _keyNameName: j[1]}
            res1.append(t)

    res[_keyNameDistrict] = res1
    return jsonify(res), 200


@app.route('/get_dist_value')
def get_dist_for_dist_id():
    dist_id = request.args.getlist(_queryParamDistId)
    base_url = 'outputs/datasets/'
    file = 'new_places.csv'
    df = pd.read_csv(base_url + file)
    if len(dist_id) == 1:
        dist_id = dist_id[0].split(',')
        dist_id = [int(d) for d in dist_id]
    print(f'/get_dist_value endpoint called with dist_id={dist_id}')
    if dist_id == [1000, ]:
        return jsonify({_keyNameDists: ['Test', ]})
    dists = list(df['District'])
    res = []
    for d in dist_id:
        res.append(dists[d])

    print(f'/get_dist_value endpoint returned => {res}')

    return jsonify({_keyNameDists: res}), 200


@app.route('/get_seasons')
def get_seasons():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    print(f'/get_types_of_crops endpoint called with state={state} and '
          f'dist={dist}')
    if state == 'Test' and dist == 'Test':
        return jsonify({_keyNameSeasons: ['Test', ]})
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/yield/'
    file = 'found1_all_18.csv'
    df = pd.read_csv(base_url + file)
    df1 = df[df['State'] == state]
    df1 = df1[df1['District'] == dist]
    seasons = []
    if df1.shape[0] > 0:
        seasons = list(df1['Season'].unique())

    return jsonify({_keyNameSeasons: seasons}), 200


@app.route('/get_crops')
def get_crops():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    season = request.args.get(_queryParamSeason)
    print(f'/get_crops endpoint called with state={state}, '
          f'dist={dist} and season={season}')
    if state == 'Test' and dist == 'Test' and season == 'Test':
        return jsonify({_keyNameCrops: [{_keyNameCropId: 'Test', _keyNameName: 'Test', }, ]})
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/yield/'
    file = 'found1_all_18.csv'
    df = pd.read_csv(base_url + file)
    all_crops = list(df['Crop'].unique())
    all_crops_dict = {}
    for i, crop in enumerate(all_crops):
        all_crops_dict[crop] = str(i)

    df1 = df[df['State'] == state]
    df1 = df1[df1['District'] == dist]
    df1 = df1[df1['Season'] == season]
    crops_res = []
    if df1.shape[0] > 0:
        crops = list(df1['Crop'].unique())
        for crop in crops:
            crops_res.append({_keyNameCropId: all_crops_dict[crop], _keyNameName: crop, })

    return jsonify({_keyNameCrops: crops_res}), 200


@app.route('/get_seasons_v2')
def get_seasons_v2():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    crop = request.args.get(_queryParamCrop)
    print(f'/get_types_of_crops endpoint called with state={state} and '
          f'dist={dist} and crop={crop}')
    if state == 'Test' and dist == 'Test' and crop == 'Test':
        return jsonify({_keyNameSeasons: ['Test', ]})
    base_url = 'outputs/datasets/'
    file = 'found_crop_data.csv'
    df = pd.read_csv(base_url + file)

    df1 = df[df['State'] == state]
    df1 = df1[df1['District'] == dist]

    seasons = []
    if df1.shape[0] > 0:
        for i, j in df1.iterrows():
            if str(j[4]) == crop:
                seasons.append(j[2])

        seasons = list(set(seasons))

    return jsonify({_keyNameSeasons: seasons}), 200


@app.route('/get_crops_v2')
def get_crops_v2():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    print(f'/get_crops endpoint called with state={state}, '
          f'and dist={dist}')
    if state == 'Test' and dist == 'Test':
        return jsonify({_keyNameCrops: [{_keyNameCropId: 'Test', _keyNameName: 'Test', }, ]})
    base_url = 'outputs/datasets/'
    file = 'found_crop_data.csv'
    df = pd.read_csv(base_url + file)

    df1 = df[df['State'] == state]
    df1 = df1[df1['District'] == dist]

    crops_res = []
    if df1.shape[0] > 0:
        for i, j in df1.iterrows():
            crops_res.append({_keyNameCropId: str(j[4]), _keyNameName: j[3]})

        crops_res = list({v[_keyNameCropId]: v for v in crops_res}.values())

    return jsonify({_keyNameCrops: crops_res}), 200


@app.route('/yield')
def predict_yield():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    season = request.args.get(_queryParamSeason)
    crop = request.args.get(_queryParamCrop)

    state = state.replace(' ', '+')
    dist = dist.replace(' ', '+')
    if state is None or dist is None or season is None or crop is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    print(f'/yield endpoint called with state={state}, '
          f'dist={dist}, season={season} and crop={crop}')
    if state == 'Test' and dist == 'Test' and season == 'Test' and crop == 'Test':
        return jsonify({_keyNameYield: [1.0, ]})
    files = os.listdir('outputs/yield')

    file = dist + ',' + state + ',' + season + ',' + crop + '.csv'
    try:
        if file not in files:
            # yield_caller(state, dist, season, crop)
            return jsonify({'message': 'The requested location cannot be processed'}), 404

        print(f'All yield prediction complete for state={state}, dist={dist}'
              f', season={season} and crop={crop}')

        df1 = pd.read_csv(f'outputs/yield/{file}')

        my_values = {
            _keyNameYield: df1['Predicted'].to_list(),
        }

        return jsonify(my_values), 200

    except FileNotFoundError:
        return jsonify({'message': 'The requested location cannot be processed'}), 404


@app.route('/statistics_data')
def generate_statistics_data():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)

    state = state.replace(' ', '+')
    dist = dist.replace(' ', '+')

    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    print(f'/statistics_data endpoint called with state={state} and '
          f'dist={dist}')

    if state == 'Test' and dist == 'Test':
        return jsonify({_keyNameTemperature: [{'y1': 'Test'}], _keyNameHumidity: [{'y1': 'Test'}],
                        _keyNameRainfall: [{'y1': 'Test'}, ]})
    res = {}
    try:
        rain = fetch_rainfall_whole_data(state, dist)
        temp = fetch_temp_whole_data(state, dist)
        humidity = fetch_humidity_whole_data(state, dist)

        res[_keyNameTemperature] = temp
        res[_keyNameHumidity] = humidity
        res[_keyNameRainfall] = rain
    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    return jsonify(res), 200

# Uncomment when running locally
# app.run(port=4999)

# Uncomment when pushing to GCP
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=4999, debug=True)
