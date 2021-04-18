from flask import jsonify, request

from routes.utils.key_names import KeyNames
from statistics_data_fetcher import fetch_rainfall_whole_data, fetch_temp_whole_data, fetch_humidity_whole_data
from . import routes


@routes.route('/statistics_data')
def generate_statistics_data():
    state = request.args.get(KeyNames.queryParamState)
    dist = request.args.get(KeyNames.queryParamDist)

    state = state.replace(' ', '+')
    dist = dist.replace(' ', '+')

    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    print(f'/statistics_data endpoint called with state={state} and '
          f'dist={dist}')

    if state == 'Test' and dist == 'Test':
        return jsonify({KeyNames.keyNameTemperature: [{'y1': 'Test'}], KeyNames.keyNameHumidity: [{'y1': 'Test'}],
                        KeyNames.keyNameRainfall: [{'y1': 'Test'}, ]})
    res = {}
    try:
        rain = fetch_rainfall_whole_data(state, dist)
        temp = fetch_temp_whole_data(state, dist)
        humidity = fetch_humidity_whole_data(state, dist)

        res[KeyNames.keyNameTemperature] = temp
        res[KeyNames.keyNameHumidity] = humidity
        res[KeyNames.keyNameRainfall] = rain
    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    return jsonify(res), 200
