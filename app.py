import os
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from humidity_predictions import humidity_caller
from rainfall_predictions import rain_caller
from temp_predictions import temperature_caller
from weather_filters import multiple_states

app = Flask(__name__)
auth = HTTPBasicAuth()
CORS(app)

users = {
    "sughosh": generate_password_hash("hello")
}


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


@app.route('/')
def home():
    return 'Hello World'


@app.route('/weather')
def weather():
    state = request.args.get('state')
    dist = request.args.get('dist')
    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404
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

        df1 = pd.read_csv(f'outputs/temp/{file}')
        df2 = pd.read_csv(f'outputs/humidity/{file}')
        df3 = pd.read_csv(f'outputs/rainfall/{file}')

        my_values = {
            'temperature': df1['Predicted'].to_list(),
            'humidity': df2['Predicted'].to_list(),
            'rainfall': df3['Predicted'].round(2).to_list()
        }

        return jsonify(my_values), 200

    except FileNotFoundError:
        return jsonify({'message': 'The requested location cannot be processed'}), 404


@app.route('/weather1/<string:state>/<string:dist>')
def weather1(state, dist):
    files1 = os.listdir('outputs/temp')

    file = dist + ',' + state + '.csv'
    if file in files1:
        df1 = pd.read_csv(f'outputs/humidity/{file}')

        my_values = {
            'humidity': df1['Predicted'].to_list(),

        }

        return jsonify(my_values), 200

    else:
        try:
            humidity_caller(state, dist)

            df1 = pd.read_csv(f'outputs/humidity/{file}')

            my_values = {
                'humidity': df1['Predicted'].to_list(),

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
    dists = request.args.get('dists')
    years = request.args.getlist('years')
    params = request.args.getlist('params')

    if len(states) == 1:
        states = states[0].split(',')
    if len(dists) == 1:
        dists = dists[0].split(',')
    if len(years) == 1:
        years = years[0].split(',')
        years = [int(i) for i in years]
    if len(params) == 1:
        params = params[0].split(',')

    if len(states) > 1:
        multiple_states(states, years, params)
        handle = ZipFile('required_downloads.zip', 'w')
        if 'temp' in params:
            handle.write('filter_outputs/weather/temp.csv', 'temperature.csv', compress_type=ZIP_DEFLATED)
        if 'humidity' in params:
            handle.write('filter_outputs/weather/humidity.csv', 'humidity.csv', compress_type=ZIP_DEFLATED)
        if 'rainfall' in params:
            handle.write('filter_outputs/weather/rain.csv', 'rainfall.csv', compress_type=ZIP_DEFLATED)
        handle.close()

        return send_from_directory('', 'required_downloads.zip', as_attachment=True), 200

    return jsonify({'message': 'The requested location cannot be processed'}), 404


@app.route('/weather/file1')
@auth.login_required
def download_temp_file():
    state = request.args.get('state')
    dist = request.args.get('dist')
    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    file = f'{dist},{state}.csv'
    if file in os.listdir('outputs/temp'):
        return send_from_directory('outputs/temp', f'{dist},{state}.csv', as_attachment=True)
    else:
        return jsonify({'message': 'File not found'}), 404


@app.route('/weather/file2')
@auth.login_required
def download_humidity_file():
    state = request.args.get('state')
    dist = request.args.get('dist')
    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    file = f'{dist},{state}.csv'
    if file in os.listdir('outputs/humidity'):
        return send_from_directory('outputs/humidity', f'{dist},{state}.csv', as_attachment=True)
    else:
        return jsonify({'message': 'File not found'}), 404


@app.route('/weather/file3')
@auth.login_required
def download_rainfall_file():
    state = request.args.get('state')
    dist = request.args.get('dist')
    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    file = f'{dist},{state}.csv'
    if file in os.listdir('outputs/rainfall'):
        return send_from_directory('outputs/rainfall', f'{dist},{state}.csv', as_attachment=True)
    else:
        return jsonify({'message': 'File not found'}), 404


@app.route('/weather/files')
@auth.login_required
def download_files():
    state = request.args.get('state')
    dist = request.args.get('dist')
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

        return send_from_directory('', f'{dist},{state}.zip', as_attachment=True)
    else:
        return jsonify({'message': 'File not found'}), 404


def preprocessing(s):
    s = s.replace('+', ' ')
    s = s.capitalize()
    return s


@app.route('/get_states')
def get_state():
    base_url = 'https://raw.githubusercontent.com/bssughosh/agri-guide-data/master/datasets/weather/'
    file = 'places.csv'
    df = pd.read_csv(base_url + file)

    df['State'] = df['State'].apply(lambda c: preprocessing(c))
    res = {}
    res1 = []

    states = list(df['State'].unique())

    for i, j in enumerate(states):
        t = {'id': str(i + 1), 'name': j}
        res1.append(t)

    res['state'] = res1
    return jsonify(res), 200


@app.route('/get_dists')
def get_dist():
    state_id = request.args.get('state_id')
    if state_id is None:
        return jsonify({'message': 'State ID not found'}), 404
    try:
        state_id = int(state_id)
    except ValueError:
        return jsonify({'message': 'State ID not found'}), 404

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
            t = {'id': str(i), 'state_id': str(k), 'name': j[1]}
            res1.append(t)

    res['district'] = res1
    return jsonify(res), 200


app.run(port=4999)
