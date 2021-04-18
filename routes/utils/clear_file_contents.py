def clear_file_contents(param):
    if param == 'yield':
        file = open(f'filter_outputs/yield/{param}.csv', 'w+')
        file.close()
    else:
        file = open(f'filter_outputs/weather/{param}.csv', 'w+')
        file.close()
