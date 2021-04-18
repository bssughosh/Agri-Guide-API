from flask import Blueprint

routes = Blueprint('routes', __name__)

# Default URL
from .home import *

# Prediction Endpoints
from .weather_prediction import *
from .yield_prediction import *

# Downloads Endpoints
from .weather_downloads import *
from .downloads import *

# Utility Endpoints
from .get_states import *
from .get_state_for_state_id import *
from .get_dists import *
from .get_dist_for_dist_id import *
from .get_crops import *
from .get_seasons import *

# Statistics Endpoints
from .weather_statistics import *
from .yield_statistics import *
