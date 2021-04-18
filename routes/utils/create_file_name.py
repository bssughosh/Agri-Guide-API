def create_file_name():
    present_time = datetime.now()
    _day = present_time.day
    _month = present_time.month
    _year = present_time.year
    _minute = present_time.minute
    _hour = present_time.hour
    _second = present_time.second

    _filename = str(_year) + '_' + str(_month) + '_' + str(_day) + '_' + str(_hour) + '_' + str(_minute) + '_' + str(
        _second) + '.zip'

    return _filename
