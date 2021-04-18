from zipfile import ZipFile, ZIP_DEFLATED

from flask import jsonify, request, send_file

from routes.utils.key_names import KeyNames
from weather_filters import multiple_states, single_loc, multiple_dists
from yield_filters import multiple_states_yield, single_loc_yield, multiple_dists_yield
from . import routes
from .utils.clear_file_contents import clear_file_contents
from .utils.create_file_name import create_file_name


@routes.route('/agri_guide/downloads')
def download_with_filters():
    """
    states: List of states\n
    dists: Will be used only when len(states) == 1\n
    years: If years == 0 then all years else will accept length of 2\n
    params: temp,humidity,rainfall,yield\n
    :return: ZIP file containing the required CSV files
    """
    states = request.args.getlist(KeyNames.queryParamStates)
    dists = request.args.getlist(KeyNames.queryParamDists)
    years = request.args.getlist(KeyNames.queryParamYears)
    params = request.args.getlist(KeyNames.queryParamParams)
    try:
        if len(states) == 1:
            states = states[0].split(',')
            states = [state.replace(' ', '+') for state in states]
        if len(dists) == 1:
            dists = dists[0].split(',')
            dists = [dist.replace(' ', '+') for dist in dists]
        if len(years) == 1:
            years = years[0].split(',')
            years = [int(i) for i in years]
        if len(params) == 1:
            params = params[0].split(',')

        print(f'/agri_guide/downloads endpoint called with states={states}, '
              f'dists={dists}, years={years} and params={params}')

        clear_file_contents('temp')
        clear_file_contents('humidity')
        clear_file_contents('rain')
        clear_file_contents('yield')

        if len(states) > 1:
            multiple_states(states, years, params)

        if len(states) == 1 and len(dists) > 1:
            multiple_dists(states[0], dists, years, params)

        if len(states) == 1 and len(dists) == 1:
            if dists == ['0']:
                multiple_dists(states[0], dists, years, params)
            else:
                single_loc(states[0], dists[0], years, params)
        try:
            if 'yield' in params:
                if len(states) > 1:
                    multiple_states_yield(states)

                if len(states) == 1 and len(dists) > 1:
                    multiple_dists_yield(states[0], dists)

                if len(states) == 1 and len(dists) == 1:
                    if dists == ['0']:
                        multiple_dists_yield(states[0], dists)
                    else:
                        single_loc_yield(states[0], dists[0])
        except:
            print(f'yield data not found for states={states} and dists={dists}')

        handle = ZipFile('required_downloads.zip', 'w')
        if 'temp' in params:
            handle.write('filter_outputs/weather/temp.csv', 'temperature.csv', compress_type=ZIP_DEFLATED)
        if 'humidity' in params:
            handle.write('filter_outputs/weather/humidity.csv', 'humidity.csv', compress_type=ZIP_DEFLATED)
        if 'rainfall' in params:
            handle.write('filter_outputs/weather/rain.csv', 'rainfall.csv', compress_type=ZIP_DEFLATED)
        if 'yield' in params:
            handle.write('filter_outputs/yield/yield.csv', 'yield.csv', compress_type=ZIP_DEFLATED)
        handle.close()

        print(f'ZipFile created for states={states}, '
              f'dists={dists}, years={years} and params={params}')

        response = send_file('required_downloads.zip', as_attachment=True, attachment_filename=create_file_name())

        return response, 200

    except:
        return jsonify({'message': 'The requested location cannot be processed'}), 404
