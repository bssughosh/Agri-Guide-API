from . import routes


@routes.route('/')
def home():
    print(f'/home endpoint called ')
    return 'Agri Guide'
