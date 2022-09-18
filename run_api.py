import sys
sys.path.insert(0, '../')
from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from apiClass.dept_dest_code_api import DestCode, GetDest
from apiClass.extract_odPairs_api import ExtractODPairs
from apiClass.future_track_api import FutureTrackApi
from apiClass.port_dictionary_api import PortDict
#from apiClass.get_all_total_dest_code import GetDest

# Create a new Flask app
app = Flask(__name__)
CORS(app)
api = Api(app)

# API 리스트
api.add_resource(FutureTrackApi, '/getTrack')
api.add_resource(ExtractODPairs, '/getODPairs')
api.add_resource(DestCode, '/destCode')
api.add_resource(PortDict, '/portDict')
api.add_resource(GetDest, '/getDest')

# # set host address
HOST = '0.0.0.0'
#HOST = '127.0.0.1'
#HOST = '43.200.0.13'
PORT = 27740
debugMode = False

if __name__ == "__main__":
    # run for any ip address
    app.run(host=HOST, port=PORT, debug=debugMode)

