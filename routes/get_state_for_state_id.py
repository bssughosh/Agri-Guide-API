import pandas as pd
from flask import jsonify, request

from routes.utils.key_names import KeyNames
from . import routes


@routes.route('/get_state_value')
def get_state_for_state_id():
    state_id = request.args.getlist(KeyNames.queryParamStateId)
    base_url = 'outputs/datasets/'
    file = 'new_places.csv'
    df = pd.read_csv(base_url + file)
    if len(state_id) == 1:
        state_id = state_id[0].split(',')
        state_id = [(int(s) - 1) for s in state_id]
    print(f'/get_state_value endpoint called with state_id={state_id}')
    if state_id == [1000, ]:
        return jsonify({KeyNames.keyNameStates: ['Test', ]})
    states = list(df['State'].unique())
    res = []
    for s in state_id:
        res.append(states[s])

    print(f'/get_state_value endpoint returned => {res}')

    return jsonify({KeyNames.keyNameStates: res}), 200
