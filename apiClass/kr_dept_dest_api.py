from lib.postgreSQL import extract_data_from_postgresQL, CONST_KR_TABLE
from flask_restful import Resource, reqparse
from flask import make_response
from datetime import datetime
import pandas as pd
import warnings
from apiClass.dept_dest_code_api import MODEL, DF_KR_SHIP_DICTIONARY, CONST_CUR_TABLE, CONST_KR_MAPING_TABLE,\
                                        DF_SHORT_DICTIONARY, TOKEN
from predict_dest_code import main, insert_into_table
warnings.filterwarnings(action='ignore')



class KrDeptDest(Resource):

    def get(self):

        parser = reqparse.RequestParser()  # initialize

        # parser.add_argument('token', required=True, help="Token cannot be blank!")  # add args
        parser.add_argument('token', required=False, help="Token cannot be blank!")

        args = parser.parse_args()  # parse arguments to dictionary

        token = args['token']

        ship_imo = tuple(DF_KR_SHIP_DICTIONARY['IMO No,'])
        COL = "imo"
        cur_records, past_records = extract_data_from_postgresQL(ship_imo, CONST_CUR_TABLE, \
                                                                     CONST_KR_MAPING_TABLE, COL)
        df_final_data = main(cur_records, past_records, DF_SHORT_DICTIONARY, MODEL)
        insert_into_table(df_final_data, CONST_KR_TABLE)



        if token in TOKEN:

            # 파일로 다운로드하는 api
            dicts = {"result": "ok"}
            dicts["type"] = "File"
            data = df_final_data.to_csv()
            response = make_response(data)
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = 'attachment; filename=kr_ais_data.csv'

            return response
        else:
            return {"Message": "You are not authorized!!!"}, 200