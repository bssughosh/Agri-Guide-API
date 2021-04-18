import pandas as pd
from flask import jsonify, request

from routes.utils.key_names import KeyNames
from . import routes


@routes.route('/get_crops')
@routes.route('/get_crops_v2')
def get_crops():
    state = request.args.get(KeyNames.queryParamState)
    dist = request.args.get(KeyNames.queryParamDist)
    print(f'/get_crops endpoint called with state={state}, '
          f'and dist={dist}')
    if state == 'Test' and dist == 'Test':
        return jsonify({KeyNames.keyNameCrops: [{KeyNames.keyNameCropId: 'Test', KeyNames.keyNameName: 'Test', }, ]})
    base_url = 'outputs/datasets/'
    file = 'found_crop_data.csv'
    df = pd.read_csv(base_url + file)

    df1 = df[df['State'] == state]
    df1 = df1[df1['District'] == dist]

    crops_res = []
    if df1.shape[0] > 0:
        for i, j in df1.iterrows():
            crops_res.append({KeyNames.keyNameCropId: str(j[4]), KeyNames.keyNameName: j[3]})

        crops_res = list({v[KeyNames.keyNameCropId]: v for v in crops_res}.values())

    return jsonify({KeyNames.keyNameCrops: crops_res}), 200
