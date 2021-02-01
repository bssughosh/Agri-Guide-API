import os
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from humidity_predictions import humidity_caller
from rainfall_predictions import rain_caller
from statistics_data_fetcher import fetch_rainfall_whole_data, fetch_temp_whole_data, fetch_humidity_whole_data
from temp_predictions import temperature_caller
from weather_filters import multiple_states, single_loc, multiple_dists
from yield_prediction import yield_caller

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
    state = request.args.get('state')
    dist = request.args.get('dist')
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
            temperature_caller(state, dist)

        if file not in files2:
            humidity_caller(state, dist)

        if file not in files3:
            rain_caller(state, dist)

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


@app.route('/weather/downloads')
def download_weather_filters():
    """
    states: List of states\n
    dists: Will be used only when len(states) == 1\n
    years: If years == 0 then all years else will accept length of 2\n
    params: temp,humidity,rainfall\n
    :return: ZIP file containing the required CSV files
    """
    states = request.args.getlist('states')
    dists = request.args.getlist('dists')
    years = request.args.getlist('years')
    params = request.args.getlist('params')
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

        return send_from_directory('', 'required_downloads.zip', as_attachment=True), 200

    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404


@app.route('/weather/files')
@auth.login_required
def download_weather_predicted_files():
    state = request.args.get('state')
    dist = request.args.get('dist')
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
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = 'places.csv'
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
    state_id = request.args.getlist('state_id')
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = 'places.csv'
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
    state_id = request.args.get('state_id')
    if state_id is None:
        return jsonify({'message': 'State ID not found'}), 404
    try:
        state_id = int(state_id)
        if state_id == 1000:
            return jsonify({_keyNameDistrict: [{_keyNameId: 'Test', _keyNameStateId: 'Test', _keyNameName: 'Test'}, ]})
    except ValueError:
        return jsonify({'message': 'State ID not found'}), 404

    print(f'/get_dists endpoint called with state_id={state_id}')

    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = 'places.csv'
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
    dist_id = request.args.getlist('dist_id')
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = 'places.csv'
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
    state = request.args.get('state')
    dist = request.args.get('dist')
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
    state = request.args.get('state')
    dist = request.args.get('dist')
    season = request.args.get('season')
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


@app.route('/yield')
def predict_yield():
    state = request.args.get('state')
    dist = request.args.get('dist')
    season = request.args.get('season')
    crop = request.args.get('crop')

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
            yield_caller(state, dist, season, crop)

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
    state = request.args.get('state')
    dist = request.args.get('dist')

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


app.run(port=4999)
