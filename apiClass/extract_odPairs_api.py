import sys
sys.path.insert(0, '../')
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask_restful import Resource, reqparse
from lib.kr_extract_from_mongodb import get_history_data, main
from lib.postgreSQL import *

class ExtractODPairs(Resource):

    def post(self):

        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('mmsi', type=float, required=True, help="Mmsi cannot be blank!")
        parser.add_argument('token', required=True, help="Token cannot be blank!")

        args = parser.parse_args()  # parse arguments to dictionary

        mmsi = str(int(args['mmsi']))
        print("mmsi:", mmsi)
        token = args['token']

        today = datetime.now()
        #before = today - relativedelta(months=3)
        before = datetime(2022, 9, 1)
        todate = today.strftime("%Y-%m-%d")
        #todate = "2022-06-30'
        fromdate = before.strftime("%Y-%m-%d")
        records = [get_history_data(mmsi, fromdate, todate)]

        # 토큰 가져오기
        table_name = "member"
        token_conn, token_curs, token_df = connect_DB3(table_name)
        TOKEN = list(token_df['token'])
        disconnection(token_conn, token_curs)

        if records[0].shape[0]==0:
            
            return {"Message": "Data is empty!!!"}, 200

        elif token in TOKEN and records[0].shape[0]!=0:

            dicts = {"result": "ok"}
            dicts["type"] = "PortInfo"
            data = {}
            data['list'] = []
            sailing_records = main(records)
            sailing_records = sailing_records[::-1]
            
            for i in range(len(sailing_records)):
                # idx = len(sailing_records[i]['nav_info']['dt_pos_utc']) - 1
                data['list'].append({'portCode': sailing_records[i]['origin']['port_code'],\
                        'arrival_date': sailing_records[i]['nav_info']['dt_pos_utc'][0].strftime('%Y-%m-%d')})
                if i == 4:
                    break
            #dicts['data'] = data
            
            if len(data)>0:
                dicts['data'] = data
                
            else:
                return {"Message": "OD pairs is empty!!!"}, 200

            return dicts, 200
            
        else:
            return {"Message": "You are not authorized!!!"}, 200
