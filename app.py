from flask import Flask
from flask_cors import CORS

from routes import *
from statistics_data_fetcher import fetch_yield_whole_data

app = Flask(__name__)
CORS(app)
app.register_blueprint(routes)

_keyNameTemperature = 'temperature'
_keyNameHumidity = 'humidity'
_keyNameRainfall = 'rainfall'
_keyNameDists = 'dists'
_keyNameDistrict = 'district'
_keyNameState = 'state'
_keyNameStates = 'states'
_keyNameId = 'id'
_keyNameStateId = 'state_id'
_keyNameSeasons = 'seasons'
_keyNameCrops = 'crops'
_keyNameYield = 'yield'
_keyNameName = 'name'
_keyNameCropId = 'crop_id'

_queryParamState = 'state'
_queryParamDist = 'dist'
_queryParamStates = 'states'
_queryParamDists = 'dists'
_queryParamYears = 'years'
_queryParamParams = 'params'
_queryParamStateId = 'state_id'
_queryParamDistId = 'dist_id'
_queryParamSeason = 'season'
_queryParamCrop = 'crop'




@app.route('/yield-statistics')
def get_statistics_for_crop():
    state = request.args.get(_queryParamState)
    dist = request.args.get(_queryParamDist)
    season = request.args.get(_queryParamSeason)
    crop = request.args.get(_queryParamCrop)

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
        res[_keyNameYield] = yield_res

    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404

    return jsonify(res), 200


# Uncomment when running locally
app.run(port=4999)

# Uncomment when pushing to GCP
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=4999, debug=True)
