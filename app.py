from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from routes import *
from statistics_data_fetcher import fetch_rainfall_whole_data, fetch_temp_whole_data, fetch_humidity_whole_data, \
    fetch_yield_whole_data

app = Flask(__name__)
auth = HTTPBasicAuth()
CORS(app)
app.register_blueprint(routes)

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


@app.route('/yield-statistics')
def get_statistics_for_crop():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    season = request.args.get(_queryParamSeason)
    crop = request.args.get(_queryParamCrop)

    print(f'/yield-statistics endpoint called with state={state}, '
          f'dist={dist}, crop={crop} and season={season}')

    base_url = 'outputs/datasets/'
    file = 'all_crops.csv'
    crop_data = pd.read_csv(base_url + file)

    crop_name = ''

    for i, j in crop_data.iterrows():
        if int(j[1]) == int(crop):
            crop_name = j[0]
            break

    res = {}
    try:
        yield_res = fetch_yield_whole_data(state, dist, crop_name, season)
        res[_keyNameYield] = yield_res

    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    return jsonify(res), 200


# Uncomment when running locally
app.run(port=4999)

# Uncomment when pushing to GCP
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=4999, debug=True)
