import pandas as pd
from flask import jsonify, request

from routes.utils.key_names import KeyNames
from statistics_data_fetcher import fetch_yield_whole_data
from . import routes


@routes.route('/yield-statistics')
def get_statistics_for_crop():
    state = request.args.get(KeyNames.queryParamState)
    dist = request.args.get(KeyNames.queryParamDist)
    season = request.args.get(KeyNames.queryParamSeason)
    crop = request.args.get(KeyNames.queryParamCrop)

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
        res[KeyNames.keyNameYield] = yield_res

    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    return jsonify(res), 200
