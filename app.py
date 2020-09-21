from flask import Flask, jsonify, send_from_directory
import pandas as pd
import os
from zipfile import ZipFile, ZIP_DEFLATED
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

from humidity_predictions import humidity_caller
from temp_predictions import temperature_caller
from rainfall_predictions import rain_caller

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


@app.route('/weather/<string:state>/<string:dist>')
def weather(state, dist):
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
        df1 = pd.read_csv(f'outputs/temp/{file}')

        my_values = {
            'temperature': df1['Predicted'].to_list(),

        }

        return jsonify(my_values), 200

    else:
        try:
            temperature_caller(state, dist)

            df1 = pd.read_csv(f'outputs/temp/{file}')

            my_values = {
                'temperature': df1['Predicted'].to_list(),

            }
            return jsonify(my_values), 200
        except FileNotFoundError:
            return jsonify({'message': 'The requested location cannot be processed'}), 404


@app.route('/weather/file1/<string:state>/<string:dist>')
@auth.login_required
def download_temp_file(state, dist):
    file = f'{dist},{state}.csv'
    if file in os.listdir('outputs/temp'):
        return send_from_directory('outputs/temp', f'{dist},{state}.csv', as_attachment=True)
    else:
        return jsonify({'message': 'File not found'}), 404


@app.route('/weather/file2/<string:state>/<string:dist>')
@auth.login_required
def download_humidity_file(state, dist):
    file = f'{dist},{state}.csv'
    if file in os.listdir('outputs/humidity'):
        return send_from_directory('outputs/humidity', f'{dist},{state}.csv', as_attachment=True)
    else:
        return jsonify({'message': 'File not found'}), 404


@app.route('/weather/file3/<string:state>/<string:dist>')
@auth.login_required
def download_rainfall_file(state, dist):
    file = f'{dist},{state}.csv'
    if file in os.listdir('outputs/rainfall'):
        return send_from_directory('outputs/rainfall', f'{dist},{state}.csv', as_attachment=True)
    else:
        return jsonify({'message': 'File not found'}), 404


@app.route('/weather/files/<string:state>/<string:dist>')
@auth.login_required
def download_files(state, dist):
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


# app.run(port=4999)
