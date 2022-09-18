import sys
sys.path.insert(0, '../')
import re
from apiClass.dept_dest_code_api import DF_FINAL_DICTIONARY, TOKEN
from flask_restful import Resource, reqparse


class PortDict(Resource):

    def post(self):

        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('token', required=True, help="Token may be blank!")

        # parser.add_argument('version')
        args = parser.parse_args()  # parse arguments to dictionary
        portCodes = list(DF_FINAL_DICTIONARY['port_code'])

        token = args['token']


        if token in TOKEN:

            dicts = {"result": "ok"}
            dicts["type"] = "PortDictInfo"
            data = {}
            data['list'] = []
            for port in portCodes:
                port = re.sub("[^a-zA-Z0-9]", '', port).lstrip().rstrip()
                data['list'].append(port)
            dicts['data'] = data
            return dicts, 200  # return data and 200 OK code

        else:
            return {"Message": "You are not authorized!!!"}, 200





