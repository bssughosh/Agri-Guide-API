# Agri-Guide-API

## API Endpoints - 
-----

1. Weather
    * **/weather/[state]/[dist]** : JSON output containing temperature, humidity and rainfall predictions for 2020
    * **/weather/file1/[state]/[dist]** : Download the predicted _temperature_ values in CSV format
    * **/weather/file2/[state]/[dist]** : Download the predicted _humidity_ values in CSV format
    * **/weather/file3/[state]/[dist]** : Download the predicted _rainfall_ values in CSV format
    * **/weather/files/[state]/[dist]** : Download _all_ the predicted values in CSV format compressed together in a ZIP file
    
