import pickle
import pandas as pd
from flask_restful import Resource, reqparse
from ..predict_dest_code import hybrid_classification, load_model, isValidDest
from ..lib.postgreSQL import *


# tableName
whatDate = datetime.now()
yearMonthDay = whatDate.strftime('%Y%m%d')
CONST_CUR_TABLE = 'lvi_prm_' + yearMonthDay + '%'
CONST_KR_MAPING_TABLE = 'kr_lvi_prm' + '%'

# PATH

CONST_OUT_FILE_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/out/"
CONST_IN_FILE_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/in/"
CONST_PICKLE_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/pickle/"
CONST_SAVED_MODEL_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/saved_model/"

"""
# PATH
CONST_OUT_FILE_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/out/"
CONST_IN_FILE_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/in/"
CONST_PICKLE_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/pickle/"
CONST_SAVED_MODEL_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/saved_model/"
"""

# TRAIN DATA DICTIONARY
DF_SHORT_DICTIONARY = pd.read_csv(CONST_IN_FILE_PATH + "short_port_code_dictionary_v7.csv", sep=',', encoding='latin-1', keep_default_na=False)
DF_FINAL_DICTIONARY = pd.read_csv(CONST_IN_FILE_PATH + "final_port_code_dictionary.csv", sep=',',  encoding='latin-1')
DF_KR_SHIP_DICTIONARY = pd.read_csv(CONST_IN_FILE_PATH + "kr_ship_1877.csv", sep=',',  encoding='utf-8')

# MACHINE LEARNING DATASET
CONS_FINAL_FUTURE_TRACK = CONST_SAVED_MODEL_PATH + "final_pickle/"
CONS_REPRE_FUTURE_TRACK = CONST_SAVED_MODEL_PATH + "representative_pickle/"
CONST_SAVED_MODEL = CONST_SAVED_MODEL_PATH + "my_model_short_rnn_v7_tf2.8"
KR_CONST_PICKLE_FILE = CONST_IN_FILE_PATH + "kr_dest_reg_data.pickle"

# LSTM Model 로드
MODEL = load_model(CONST_SAVED_MODEL)

# 국가코드에서 국가이름 매핑 사전
with open(CONST_IN_FILE_PATH+"cnty_dicts.pickle", 'rb') as lf:
    COUNTRY_DICTIONARY = pickle.load(lf)


# 토큰 가져오기
table_name = "member"
token_conn, token_curs, token_df = connect_DB3(table_name)
TOKEN = list(token_df['token'])
disconnection(token_conn, token_curs)


class GetDest(Resource):

    def post(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('destination', type=str, required=True, action='append', help="Destination cannot be blank!")
        # parser.add_argument('token', required=True, help="Token cannot be blank!")

        args = parser.parse_args()  # parse arguments to dictionary
        destLists = args['destination']
        # token = args['token']
        dept_dest_code = []
        for i in range(len(destLists)):
            tmp_dest = destLists[i]
            isValid, cleaned_dest = isValidDest(tmp_dest)
            dept_dest_code.append(cleaned_dest)

        predictions, percentages = hybrid_classification(dept_dest_code, MODEL, DF_SHORT_DICTIONARY,
                                                         DF_SHORT_DICTIONARY)

        dicts = {"result": "ok"}
        data = {}
        # countries = [COUNTRY_DICTIONARY[predictions[0][:2]], COUNTRY_DICTIONARY[predictions[1][:2]]]

        data['code'] = predictions
        dicts['data'] = data

        return dicts, 200  # return data and 200 OK code

class DestCode(Resource):

    def post(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('mmsi', required=True, help="Mmsi cannot be blank!")
        parser.add_argument('departure', required=False, help="departure cannot be blank!")  # add args
        parser.add_argument('destination', required=True, help="Destination cannot be blank!")
        parser.add_argument('token', required=True, help="Token cannot be blank!")

        args = parser.parse_args()  # parse arguments to dictionary
        mmsi = int(args['mmsi'])
        departure = args['departure']
        destination = args['destination']
        token = args['token']

        if departure is not None:
            dept_dest_code = [departure, destination]
            if len(departure) == 0:
                dept_dest_code = [destination]
        else:
            dept_dest_code = [destination]

        lists =[]
        for i in range(len(dept_dest_code)):
            tmp_dest = dept_dest_code[i]
            isValid, cleaned_dest = isValidDest(tmp_dest)
            lists.append(cleaned_dest)

        predictions, percentages = hybrid_classification(lists, MODEL, DF_SHORT_DICTIONARY, DF_SHORT_DICTIONARY)


        if token in TOKEN:

            dicts = {"result": "ok"}
            dicts["type"] = "DeptDestInfo"

            data = {}
            if len(predictions) == 0:
                countries = ["", COUNTRY_DICTIONARY[predictions[0][:2]]]
                print(countries)

                data['dept_code'] = {"country": "", "portCode": ""}
                data['dest_code'] = {"country": countries[1], "portCode": predictions[0]}
                dicts['data'] = data
            else:
                countries = [COUNTRY_DICTIONARY[predictions[0][:2]], COUNTRY_DICTIONARY[predictions[1][:2]]]

                data['dept_code'] = {"country": countries[0],  "portCode": predictions[0]}
                data['dest_code'] = {"country": countries[1], "portCode": predictions[1]}
                dicts['data'] = data
        
            return dicts, 200  # return data and 200 OK code
        else:

            return {"Message": "You are not authorized!!!"}, 200






