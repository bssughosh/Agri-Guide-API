import os

import pandas as pd
from flask import jsonify, request

from . import routes
from .key_names import KeyNames


@routes.route('/weather')
def weather():
    state = request.args.get(KeyNames.queryParamState)
    dist = request.args.get(KeyNames.queryParamDist)
    state = state.replace(' ', '+')
    dist = dist.replace(' ', '+')
    print(f'/weather endpoint called with state={state} and dist={dist}')
    if state is None or dist is None:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    if state == 'Test' and dist == 'Test':
        return jsonify({KeyNames.keyNameTemperature: [1, ],
                        KeyNames.keyNameHumidity: [2, ],
                        KeyNames.keyNameRainfall: [3, ]})

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
            KeyNames.keyNameTemperature: df1['Predicted'].to_list(),
            KeyNames.keyNameHumidity: df2['Predicted'].to_list(),
            KeyNames.keyNameRainfall: df3['Predicted'].round(2).to_list()
        }

        return jsonify(my_values), 200

    except FileNotFoundError:
        return jsonify({'message': 'The requested location cannot be processed'}), 404
