[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![GitHub contributors](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/contributors/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://shields.io/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

[![Anurag's github stats](https://github-readme-stats.vercel.app/api?username=bssughosh&theme=blue-green)](https://github.com/anuraghazra/github-readme-stats)

# Agri-Guide-API

## API Endpoints - 

1. General Endpoints
    * **/get_states** : JSON Output containing all the states available in the repository with its unique ID
    * **/get_dists?state_id=[state_id]** :  JSON Output containing all the districts available in the repository with a unique state ID
    * **/get_state_value?state_id={[state_id1],[state_id2]...}** : JSON Output with a list of all the names of states for the state ids passed
    * **/get_dist_value?dist_id={[dist_id1],[dist_id2]...}** : JSON Output with a list of all the names of districts for the district ids passed
    * **/agri_guide/downloads?states={[state1],[state2]...}&dists={[dist1],[dist2]...}&years={[start],[end]}&params={[temp],[humidity],[rainfall],[yield]}** : Will return a zip file containing the CSV files for the filters applied

2. Weather
   * **/weather?state=[state]&dist=[dist]** : JSON output containing temperature, humidity and rainfall predictions for
     2020
   * **/weather/files?state=[state]&dist=[dist]** : Download _all_ the predicted values in CSV format compressed
     together in a ZIP file
   * **/weather/downloads?states={[state1],[state2]...}&dists={[dist1],[dist2]...}&years={[start],[end]}&params={[temp]
     ,[humidity],[rainfall]}** : Will return a zip file containing the CSV files for the filters applied

3. Yield
   * **/get_crops?state=[state]&dist=[dist]** : JSON Output containing the possible crops for the state and dist
   * **/get_seasons?state=[state]&dist=[dist]&crop=[crop_id]** : JSON Output containing the possible seasons for a crop
     growing in the state and dist
   * **/yield?state=[state]&dist=[dist]&season=[season]&crop=[crop_id]** : JSON output containing yield predictions for
     2020 for the state and dist in a season
    
