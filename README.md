# Agri-Guide-API

## API Endpoints - 

1. General Endpoints
    * **/get_states** : JSON Output containing all the states available in the repository with its unique ID
    * **/get_dists?state_id=[state_id]** :  JSON Output containing all the districts available in the repository with a unique state ID

2. Weather
    * **/weather?state=[state]&dist=[dist]** : JSON output containing temperature, humidity and rainfall predictions for 2020
    * **/weather/file1?state=[state]&dist=[dist]** : Download the predicted _temperature_ values in CSV format
    * **/weather/file2?state=[state]&dist=[dist]** : Download the predicted _humidity_ values in CSV format
    * **/weather/file3?state=[state]&dist=[dist]** : Download the predicted _rainfall_ values in CSV format
    * **/weather/files?state=[state]&dist=[dist]** : Download _all_ the predicted values in CSV format compressed together in a ZIP file
    
