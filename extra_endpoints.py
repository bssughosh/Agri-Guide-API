# @app.route('/weather/file1')
# @auth.login_required
# def download_temp_file():
#     state = request.args.get('state')
#     dist = request.args.get('dist')
#     if state is None or dist is None:
#         return jsonify({'message': 'The requested location cannot be processed'}), 404
#
#     file = f'{dist},{state}.csv'
#     if file in os.listdir('outputs/temp'):
#         return send_from_directory('outputs/temp', f'{dist},{state}.csv', as_attachment=True)
#     else:
#         return jsonify({'message': 'File not found'}), 404
#
#
# @app.route('/weather/file2')
# @auth.login_required
# def download_humidity_file():
#     state = request.args.get('state')
#     dist = request.args.get('dist')
#     if state is None or dist is None:
#         return jsonify({'message': 'The requested location cannot be processed'}), 404
#
#     file = f'{dist},{state}.csv'
#     if file in os.listdir('outputs/humidity'):
#         return send_from_directory('outputs/humidity', f'{dist},{state}.csv', as_attachment=True)
#     else:
#         return jsonify({'message': 'File not found'}), 404
#
#
# @app.route('/weather/file3')
# @auth.login_required
# def download_rainfall_file():
#     state = request.args.get('state')
#     dist = request.args.get('dist')
#     if state is None or dist is None:
#         return jsonify({'message': 'The requested location cannot be processed'}), 404
#
#     file = f'{dist},{state}.csv'
#     if file in os.listdir('outputs/rainfall'):
#         return send_from_directory('outputs/rainfall', f'{dist},{state}.csv', as_attachment=True)
#     else:
#         return jsonify({'message': 'File not found'}), 404
