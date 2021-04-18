import os

import pandas as pd
from flask import jsonify, request

from routes.utils.key_names import KeyNames
from . import routes


@routes.route('/yield')
def predict_yield():
    state = request.args.get(KeyNames.queryParamState)
    dist = request.args.get(KeyNames.queryParamDist)
    season = request.args.get(KeyNames.queryParamSeason)
    crop = request.args.get(KeyNames.queryParamCrop)

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
            KeyNames.keyNameYield: df1['Predicted'].to_list(),
        }

        return jsonify(my_values), 200

    except FileNotFoundError:
        return jsonify({'message': 'The requested location cannot be processed'}), 404
