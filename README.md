[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![GitHub contributors](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/contributors/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://shields.io/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# Agri-Guide-API

This is the API used for the project Agri Guide. The driver code is present inside `app.py` and the routes are present
inside the folder `routes/`. The endpoints are listed below:

## API Endpoints -

### General Endpoints

#### 1. Get States

    https://agri-guide-api.herokuapp.com/get_states

JSON Output containing all the states available in the repository with its unique ID

#### 2. Get districts for a state

    https://agri-guide-api.herokuapp.com/get_dists

Params:

| Parameter | Description | Format |
|-----------|-------------|--------|
| state_id | state ID whose districts are required | `state_id` |

JSON Output containing all the districts available in the repository with a unique state ID.

#### 3. Get state name for state ID

    https://agri-guide-api.herokuapp.com/get_state_value

Params:

| Parameter | Description | Format |
|-----------|-------------|--------|
| state_id | List of state IDs separated by a comma | `state_id1`, `state_id2` |

This API endpoint would take list of state IDs to be processed and returns a list of state names.

#### 4. Get district name for district ID

    https://agri-guide-api.herokuapp.com/get_dist_value

Params:

| Parameter | Description | Format |
|-----------|-------------|--------|
| dist_id | List of district IDs separated by a comma | `dist_id1`, `dist_id2` |

This API endpoint would take list of district IDs to be processed and returns a list of district names.

#### 5. Get Crops

    https://agri-guide-api.herokuapp.com/get_crops

Params:

| Parameter | Description | Format |
|-----------|-------------|--------|
| state | The state name in lower case is passed | `state` |
| dist | The district name in lower case is passed | `dist` |

Get list of crops that grow in a district in JSON format. It would return an ID and a name for every crop.

#### 6. Get seasons

    https://agri-guide-api.herokuapp.com/get_seasons

Params:

| Parameter | Description | Format |
|-----------|-------------|--------|
| state | The state name in lower case is passed | `state` |
| dist | The district name in lower case is passed | `dist` |
| crop | Crop ID which grows in the region | `crop_id` |

Get list of seasons that a crop is grown in a district in JSON format.
