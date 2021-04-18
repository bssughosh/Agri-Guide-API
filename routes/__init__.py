from flask import Blueprint

routes = Blueprint('routes', __name__)

from .home import *
from .weather_prediction import *
from .weather_downloads import *
from .downloads import *
