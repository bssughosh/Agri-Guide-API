import pandas as pd
from flask import jsonify, request

from routes.utils.key_names import KeyNames
from . import routes


@routes.route('/get_dist_value')
def get_dist_for_dist_id():
    dist_id = request.args.getlist(KeyNames.queryParamDistId)
    base_url = 'outputs/datasets/'
    file = 'new_places.csv'
    df = pd.read_csv(base_url + file)
    if len(dist_id) == 1:
        dist_id = dist_id[0].split(',')
        dist_id = [int(d) for d in dist_id]
    print(f'/get_dist_value endpoint called with dist_id={dist_id}')
    if dist_id == [1000, ]:
        return jsonify({KeyNames.keyNameDists: ['Test', ]})
    dists = list(df['District'])
    res = []
    for d in dist_id:
        res.append(dists[d])

    print(f'/get_dist_value endpoint returned => {res}')

    return jsonify({KeyNames.keyNameDists: res}), 200
