from flask import Flask
from flask_cors import CORS

from routes import *

app = Flask(__name__)
CORS(app)
app.register_blueprint(routes)

# Uncomment when running locally
# app.run(port=4999)

# Uncomment when pushing to GCP
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=4999, debug=True)
