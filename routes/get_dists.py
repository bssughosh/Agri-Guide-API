import pandas as pd
from flask import jsonify, request

from routes.utils.key_names import KeyNames
from . import routes
from .utils.preprocessing import preprocessing


@routes.route('/get_dists')
def get_dist():
    state_id = request.args.get(KeyNames.queryParamStateId)
    if state_id is None:
        return jsonify({'message': 'State ID not found'}), 404
    try:
        state_id = int(state_id)
        if state_id == 1000:
            return jsonify({KeyNames.keyNameDistrict: [
                {KeyNames.keyNameId: 'Test', KeyNames.keyNameStateId: 'Test', KeyNames.keyNameName: 'Test'}, ]})
    except ValueError:
        return jsonify({'message': 'State ID not found'}), 404

    print(f'/get_dists endpoint called with state_id={state_id}')

    base_url = 'outputs/datasets/'
    file = 'new_places.csv'
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
            t = {KeyNames.keyNameId: str(i), KeyNames.keyNameStateId: str(k), KeyNames.keyNameName: j[1]}
            res1.append(t)

    res[KeyNames.keyNameDistrict] = res1
    return jsonify(res), 200
