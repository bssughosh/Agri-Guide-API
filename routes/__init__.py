from flask import Blueprint

routes = Blueprint('routes', __name__)

from .home import *
from .weather_prediction import *
from .weather_downloads import *
from .downloads import *
from .get_states import *
from .get_state_for_state_id import *
from .get_dists import *
from .get_dist_for_dist_id import *
from .get_crops import *
from .get_seasons import *
