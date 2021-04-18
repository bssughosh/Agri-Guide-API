import pandas as pd
from flask import jsonify, request

from routes.utils.key_names import KeyNames
from . import routes
from .utils.preprocessing import preprocessing


@routes.route('/get_states')
def get_state():
    is_test = request.args.get('isTest')
    print(f'/get_states endpoint called')
    base_url = 'outputs/datasets/'
    file = 'new_places.csv'
    df = pd.read_csv(base_url + file)

    df['State'] = df['State'].apply(lambda c: preprocessing(c))
    res = {}
    res1 = []

    states = list(df['State'].unique())

    for i, j in enumerate(states):
        t = {KeyNames.keyNameId: str(i + 1), KeyNames.keyNameName: j}
        res1.append(t)

    if is_test == 'true':
        return jsonify({KeyNames.keyNameState: [{KeyNames.keyNameId: 'Test', KeyNames.keyNameName: 'Test'}, ]})

    res[KeyNames.keyNameState] = res1
    return jsonify(res), 200
