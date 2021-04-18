import pandas as pd
from flask import jsonify, request

from routes.utils.key_names import KeyNames
from . import routes


@routes.route('/get_seasons')
@routes.route('/get_seasons_v2')
def get_seasons():
    state = request.args.get(KeyNames.queryParamState)
    dist = request.args.get(KeyNames.queryParamDist)
    crop = request.args.get(KeyNames.queryParamCrop)
    print(f'/get_types_of_crops endpoint called with state={state} and '
          f'dist={dist} and crop={crop}')
    if state == 'Test' and dist == 'Test' and crop == 'Test':
        return jsonify({KeyNames.keyNameSeasons: ['Test', ]})
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

    return jsonify({KeyNames.keyNameSeasons: seasons}), 200
