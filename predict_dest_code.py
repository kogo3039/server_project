import sys
sys.path.insert(0, "..")
from datetime import datetime
import re
import zipfile
# from collections import OrderedDict as odict
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import os
import warnings
import pickle
import psycopg2
from tqdm import tqdm
import psycopg2.extras as extras
from multiprocessing import Pool
from lib.postgreSQL import extract_data_from_postgresQL, disconnection, CONST_KR_TABLE, connect_DB, connect_DB4, update_table_in_postgresQL
import time
warnings.filterwarnings(action='ignore')


# PATH
CONST_OUT_FILE_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/out/"
CONST_IN_FILE_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/in/"
CONST_PICKLE_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/pickle/"
CONST_SAVED_MODEL_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/saved_model/"
CONST_IN_CSV_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/csv/"
CONST_IN_ZIP_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/zip/"
CONST_IN_DEST_PATH = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyapi/dest_pickle/"
"""
# PATH
CONST_OUT_FILE_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/out/"
CONST_IN_FILE_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/in/"
CONST_PICKLE_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/pickle/"
CONST_SAVED_MODEL_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/saved_model/"
CONST_IN_CSV_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/csv/"
CONST_IN_ZIP_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/zip/"
CONST_IN_DEST_PATH = "/home/ubuntu/py_Codes/pyAPI/pyapi/dest_pickle/"
"""
# TRAIN DATA DICTIONARY
DF_SHORT_DICTIONARY = pd.read_csv(CONST_IN_FILE_PATH + "short_port_code_dictionary_v7.csv", sep=',', encoding='latin-1', keep_default_na=False)
DF_FINAL_DICTIONARY = pd.read_csv(CONST_IN_FILE_PATH + "final_port_code_dictionary.csv", sep=',',  encoding='latin-1')
DF_KR_SHIP_DICTIONARY = pd.read_csv(CONST_IN_FILE_PATH + "kr_ship_1877.csv", sep=',',  encoding='utf-8')

# MACHINE LEARNING DATASET
CONS_FINAL_FUTURE_TRACK = CONST_SAVED_MODEL_PATH + "final_pickle/"
CONS_REPRE_FUTURE_TRACK = CONST_SAVED_MODEL_PATH + "representative_pickle"
CONST_SAVED_MODEL = CONST_SAVED_MODEL_PATH + "my_model_short_rnn_v7_tf2.8"
KR_CONST_PICKLE_FILE = CONST_IN_FILE_PATH + "kr_dest_reg_data.pickle"


# 국가코드에서 국가이름 매핑 사전
with open(CONST_IN_FILE_PATH+"cnty_dicts.pickle", 'rb') as lf:
    COUNTRY_DICTIONARY = pickle.load(lf)

# MACHINE LEARNING DATASET
# CONST_SAVED_MODEL = CONST_SAVED_MODEL_PATH + "my_model_short_rnn_v7_tf2.8"
# KR_CONST_PICKLE_FILE = CONST_IN_FILE_PATH + "kr_dest_reg_data.pickle"


REGEX_SEARCHES = [
    "(((open|high|opl|(out)*[ ]*at)*[ ]*(seas*))|(a*wait(ing)*))*[ ]*((f(o|0)r|to)*[ ]*order[a-z]*)*",
    "(pacific|atlantic|pax|pan)*[ ]*ocean[ ]*(ographic|apex|patriot)*",
    "(arm(ed)* *guards*)* *(ob|on *board)*",
    "(aeb|aep|asp|awp|pbg|pebg|pgb|pjs|psb|pwbg)+ *(a|b|c|d|g|p)*",
    "eopl|(east *)*opl|opl drifting|out *side|avoid typhoon|buoy|terminal|shipyard|sea( |-)*(trial|front)",
    "driftin*g for sch*edule",
]
# template = """(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
#                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s)"""
def load_model(CONST_SAVED_MODEL):
    model = tf.keras.models.load_model(CONST_SAVED_MODEL)

    return model
# Check if the destination input is valid
def isValidDest(dest):
    valid = True
    if dest is None:
        valid = False

    if (type(dest) == float):
        tmp = str(dest)
        if tmp.lower() == "nan":
            valid = False

    dest = get_clean_destination(dest)

    if str(dest).strip() == "":
        valid = False

    # If the length of the clean destination is lesser than or equal to 2, then we set it as an invalid destination
    if len(dest) < 3:
        valid = False

    return valid, dest

def getPortCodeDictionary(port_dictionary_file):
    df = pd.read_csv(port_dictionary_file, sep=',', encoding='latin-1', keep_default_na=False)

    if "port_code" in df.columns:
        df = df.rename(columns={'port_code': 'portcode'})

    return df

def get_clean_destination(dest):
    tmpdest = dest
    if tmpdest is not None:
        # Remove very long and unknown text after double quote symbol
        if len(tmpdest) > 30:
            if tmpdest.find('"') >= 0:
                tmp = tmpdest.split('"')
                tmpdest = tmp[0]

        # Remove unwanted symbol or characters
        char_to_replace = [",", ".", "-", "/", "_", "$", "#", "!", ":", "?", ";", "'",
                           "&", "^", "(", ")", "*", "[", "]", "{", "}", "%", "=", "+",
                           "-", "`", "~", "<", ">", '"', "\n", "\r\n", "\r", "\\"]


        # Get the last text after ">" greater than character if it has no "<" lesser than character (e.g. XXX <UAE>)

        if tmpdest.find('>') >= 0 and (tmpdest.find('<') == -1 or tmpdest.find('<') > tmpdest.find('>')):
            tmp = tmpdest.split('>')

            # If the length of the array is more than 2, then it has more than 1 ">" greather than characters in the text
            # Then we have to find the proper 'destination' (e.g. xxx > yyy > WA, where WA is a meaningless text, and we should get yyy instead)
            # Sometimes, it can be a destination like "country code > city code"  (e.g. CN > FOC, and we should get both of the result)

            for i in range(len(tmp)):
                tmpdest = tmp[len(tmp) - (i + 1)]
                tmpdest = tmpdest.lower()

                # TODO:
                # Make sure after the ">" character is not "OFF" word and it should be a valid word with certain length of the number of the character

                # TODO:
                # Remove meaningless texts

                # Iterate over all key-value pairs in dictionary
                for char in char_to_replace:
                    # Replace key character with value character in string
                    tmpdest = tmpdest.replace(char, " ")
                    tmpdest = " ".join(tmpdest.split())

                # If the last text has the length less than or equal to 2, then we get the last second text as the destination
                # TODO: what if there is another meaning for the last text even though it has length of 3? (E.g. XXX > OFF, YYY > )
                if len(tmpdest.replace(" ", "")) >= 3:

                    # If the current split text is not the first text, then we can check this condition
                    # TODO: what if the first text is not 2 letters of country code but it is a short form of country name? (E.g. "IND>BOM" instead of "IN>BOM" )
                    if len(tmp) == 2 and (i < len(tmp) - 1) and len(tmpdest) == 3 and len(tmp[i].strip()) == 2:
                        tmpdest = tmp[i].strip().lower() + tmpdest

                    break

        else:
            tmpdest = tmpdest.lower()
            for char in char_to_replace:
                # Replace key character with value character in string
                tmpdest = tmpdest.replace(char, " ")
                tmpdest = " ".join(tmpdest.split())

    return tmpdest

def total_fields_in_postgresQL():

    CONST_CREATE_TB_QUERY = {
        "gid": "integer NOT NULL DEFAULT nextval('lvi_prm_download_gid_seq'::regclass) PRIMARY KEY",
        "mmsi": "numeric",
        "imo": "integer",
        "vessel_name": 'character varying(254) COLLATE pg_catalog."default"',
        "callsign": 'character varying(254) COLLATE pg_catalog."default"',
        "vessel_type": 'character varying(254) COLLATE pg_catalog."default"',
        'vessel_type_code': 'integer',
        'vessel_type_cargo': 'character varying(254) COLLATE pg_catalog."default"',
        'vessel_class': 'character varying(254) COLLATE pg_catalog."default"',
        'length': 'integer',
        'width': 'integer',
        'flag_country': 'character varying(254) COLLATE pg_catalog."default"',
        'flag_code': 'integer',
        'destination': 'character varying(254) COLLATE pg_catalog."default"',
        'eta': 'character varying(254) COLLATE pg_catalog."default"',
        'draught': 'numeric',
        'longitude': 'numeric',
        'latitude': 'numeric',
        'sog': 'numeric',
        'cog': 'numeric',
        'rot': 'numeric',
        'heading': 'numeric',
        'nav_status': 'character varying(254) COLLATE pg_catalog."default"',
        'nav_status_code': 'integer',
        'source': 'character varying(254) COLLATE pg_catalog."default"',
        'ts_pos_utc': 'character varying(254) COLLATE pg_catalog."default"',
        'ts_static_utc': 'character varying(254) COLLATE pg_catalog."default"',
        'ts_insert_utc': 'character varying(254) COLLATE pg_catalog."default"',
        'dt_pos_utc': 'character varying(254) COLLATE pg_catalog."default"',
        'dt_static_utc': 'character varying(254) COLLATE pg_catalog."default"',
        'dt_insert_utc': 'character varying(254) COLLATE pg_catalog."default"',
        'vessel_type_main': 'character varying(254) COLLATE pg_catalog."default"',
        'vessel_type_sub': 'character varying(254) COLLATE pg_catalog."default"',
        'message_type': 'integer',
        'eeid': 'numeric',
        'geom': 'geometry(Point,4326)',
        'dest_code': 'character varying(254) COLLATE pg_catalog."default"',
        'dept_code': 'character varying(254) COLLATE pg_catalog."default"',
        'etd': 'character varying(254) COLLATE pg_catalog."default"'
    }

    CONST_HEADER_LVI = list(CONST_CREATE_TB_QUERY.keys())
    return CONST_HEADER_LVI

def dicitonary_matching(dest, df_port_dictionary, only_5_letters_search=True):
    dest_arr = dest.split(" ")
    # print(dest_arr)
    # exit()
    # print(df_port_code_dictionary)

    # If the dest_arr length is exactly 1 and the value has 5 characters, then we
    # can split the text as 2:3 (country code:city code) and 3:2 (city code:country code).
    # After that, we merge them with different combination
    if len(dest_arr) == 1 and len(dest_arr[0]) == 5:
        new_dest = []
        new_dest.append(dest_arr[0])    # Append the original text first

        # Treat the first 3 characters as the city code and last 2 characters as the country code
        tmp_city_code = dest_arr[0][0:3]
        tmp_country_code = dest_arr[0][3:5]
        new_dest.append(tmp_country_code+tmp_city_code)

        dest_arr = new_dest

    # If the dest_arr length is 2 and each item value is either 2 (country code) or
    # 3 (city code) characters, then we can combine them into a completed code.
    # Since we do not know the arrangement, then we merge them and form into 2
    # possibilities.
    elif len(dest_arr) == 2:
        new_dest = []
        if len(dest_arr[0]) == 2 and len(dest_arr[1]) == 3:
            new_dest.append(dest_arr[0]+dest_arr[1])
        elif len(dest_arr[0]) == 3 and len(dest_arr[1]) == 2:
            new_dest.append(dest_arr[1]+dest_arr[0])

        dest_arr = new_dest

    matched_indices_dict = []
    matched_dests = []
    isMatch = False
    match_port_code = None

    # Reverse the destination so that we prioritize the last word first
    redest_arr = np.flip(dest_arr)
    # redest_arr = dest_arr

    # Search in all loops of redest_arr
    for tmpdest in redest_arr:
        match_dest = False
        if isMatch == False and len(tmpdest) > 1:
            # If the length of the text is equal to 5, then we can search the port code first.
            # If it is a port code, we can stop after we found the port code in the dictionary
            if len(tmpdest) == 5:
                row = df_port_dictionary.loc[df_port_dictionary['portcode'] == tmpdest.upper()]
                if len(row) > 0:
                    match_port_code = row["portcode"].to_numpy()[0]
                    isMatch = True
                    # print(row, match_index)
                # If the length of destination is exactly 5 and cannot be found in the dictionary, we set it as None.
                # So that, it will return the prediction value from the trained model outside of this function
                else:
                    match_port_code = None
            elif only_5_letters_search == False:
                regexPattern = ".*"+tmpdest.lower()+".*"

                for i in range(len(df_port_dictionary.values)):
                    # If the length of the text is equal to 2, then we can search the country code only.
                    if len(tmpdest) == 2:
                        country_code = df_port_dictionary["country_code"][i]
                        if tmpdest.lower() == country_code.lower():
                            match_dest = True
                            matched_indices_dict.append(i)

                    # If the length of the text is equal to 3, then we can search the city code (and city name) only.
                    elif len(tmpdest) == 3:
                        city_code = df_port_dictionary["city_code"][i]
                        if tmpdest.lower() == city_code.lower():
                            match_dest = True
                            matched_indices_dict.append(i)

                    else:
                        # regexPattern = ".*"+tmpdest.lower()+".*"
                        short_port_dictionary = df_port_dictionary.iloc[i][["country_name", "city_name"]]
                        for info in short_port_dictionary.values:
                            # Skip the "index" column and port code
                            # print("Searching: ", regexPattern, info.lower())
                            x = re.search(regexPattern, info.lower())
                            if x is not None:
                                match_dest = True
                                matched_indices_dict.append(i)

                if match_dest:
                    matched_dests.append(tmpdest)

    if len(matched_dests) > 0:
        matched_dictionary = df_port_dictionary.iloc[matched_indices_dict]
        match_port_code = matched_dictionary["index"].values[0]
        # print("Reduce destination from ", len(redest_arr), "to", len(matched_dests))
        # print("Reduce df_port_dictionary from ", len(df_port_dictionary.values), "to", len(matched_dictionary))
        # for tmp_dest in matched_dests:
        #     for i in range(len(matched_dictionary.values)):
        #         continue
        #         search_dictionary = df_port_dictionary.iloc[matched_indices_dict]

    return match_port_code

def neural_network_classification(destinations, model, df_label_dictionary):
    # print(model.summary())
    # print(destinations)
    # exit()
    # Perform classification with a batch size
    batchSize = 1000
    totalBatch = math.ceil(len(destinations) / float(batchSize))

    predictions = []
    percentages = []
    print("Total batch: ", totalBatch)

    def is_float(value):
        try:
            float(value)
            return True
        except:
            return False
    for i in range(totalBatch):
        # if i % 100 == 0:
        print("Running batch: ", i + 1, "/", totalBatch)
        startInd = i * batchSize
        endInd = (i + 1) * batchSize
        input_data = destinations[startInd:endInd]
        # print(len(input_data))
        # exit()
        if 'JP KZU>MX MZT' in input_data:
            idx = input_data.index('JP KZU>MX MZT')
            tmp = input_data[idx]
            tmp = tmp[7:]
            input_data[idx] = tmp
        if "KOBE(JAPAN)" in input_data:
            idx = input_data.index('KOBE(JAPAN)')
            tmp = input_data[idx]
            tmp = tmp[:4]
            # print(tmp)
            # exit()
            input_data[idx] = tmp


        preds = model.predict(input_data)
        # print(preds)
        # exit()
        labels = np.argmax(preds, axis=1)
        percs = np.max(preds, axis=1)

        for j in range(len(labels)):
            # If the input is empty text, then we give an empty prediction
            # If the input length is lesser than 3, then we give an empty prediction
            if is_float(input_data[j]):
                predictions.append("")
            elif input_data[j].strip() == "" or len(input_data[j].strip()) <= 2:
                predictions.append("")
            else:
                if "portcode" in df_label_dictionary.columns:
                    predictions.append(df_label_dictionary["portcode"][int(labels[j])])
                if "port_code" in df_label_dictionary.columns:
                    predictions.append(df_label_dictionary["port_code"][int(labels[j])])

        for j in percs:
            percentages.append(round(j * 100, 2))


    return predictions, percentages

def hybrid_classification(destinations, model, df_label_dictionary, df_port_dictionary):
    
    start_time_data_prediction = time.time()
    predictions, percentages = neural_network_classification(destinations, model, df_label_dictionary)

    end_time_data_prediction = time.time()
    print("Time usage for neural network prediction: ", (end_time_data_prediction - start_time_data_prediction))

    new_predictions = []
    total_prediction = len(predictions)
    print("Dictionary Search:")
    for index, pred in enumerate(predictions):
        if (index+1) % 10000 == 0:
            print((index+1), "/", total_prediction)
        tmp_pred = None
        tmp_percentage = percentages[index]

        tmp_dest = destinations[index].strip()


        # If the destination is FOR ORDER and the probability is lower than 0.5, then we set it as empty value
        # TODO: Need to add more condition like "open sea for orders", "sea for orders", "waiting for order(s)"
        isMeaninglessText = False
        hasMeaninglessText = False
        # if tmp_dest.lower() in meaningless_texts:
        # if tmp_dest.lower() == 'for order' or tmp_dest.lower() == 'for orders' :
        for regex_pattern in REGEX_SEARCHES:
            matched = re.finditer(regex_pattern, tmp_dest.lower())
            k = [x[0] for x in matched]
            if matched is not None and len(k) > 0:
                longest_matched = max(k)
                longest_matched = longest_matched.strip()
                if len(longest_matched) == len(tmp_dest):
                    isMeaninglessText = True
                    tmp_pred = ""
                    percentages[index] = str(percentages[index]) + " (replaced)"
                    break
                elif len(longest_matched) > 0:
                    hasMeaninglessText = True

                # print(tmp_dest.lower(), isMeaninglessText, hasMeaninglessText)

        # Warning: It is too long to do the dictionary checking. So, we just check those input with low CL percentage
        # Use the dictionary to match the input
        if isMeaninglessText == False and tmp_percentage < 90:
            # Used for dictionary matching if the last word has the length of 5 where the destination is inserted with
            # other signs (i.e. = or -) rather than a greater sign (>)
            last_word = tmp_dest.split(" ")
            last_word = last_word[len(last_word)-1]

            # If the destination contains meaningless text and the probability is lower than 0.5, then we set it as empty value
            # if tmp_percentage < 50 and tmp_dest.lower().find('for order') >= 0:
            if tmp_percentage < 50 and hasMeaninglessText == True:
                tmp_pred = ""
                percentages[index] = str(percentages[index]) + " (replaced)"

            # If the probability is lesser than 0.1 (10%), then we set it as empty string
            elif tmp_percentage <= 10:
                tmp_pred = ""
                percentages[index] = str(percentages[index]) + " (replaced)"

            elif len(tmp_dest) == 5:
                tmp_pred = dicitonary_matching(tmp_dest, df_port_dictionary)

                if tmp_pred is not None:
                    percentages[index] = str(percentages[index]) + " (replaced)"

            # TODO: Do the dictionary matching for the last text which has the length of 5
            elif len(last_word) == 5:
                tmp_pred = dicitonary_matching(last_word, df_port_dictionary)

                if tmp_pred is not None:
                    percentages[index] = str(percentages[index]) + " (replaced)"



        # If we cannot find the port code from the dictionary and the probability is lower than 0.5, then we put it
        # as an empty (None) value
        # if tmp_percentage < 50 and tmp_pred is None:
        #     tmp_pred = ""
        #     percentages[index] = str(percentages[index]) + " (replaced)"

        # If the prediction is None, it means that it is not find any matched prediction in the dictionary,
        # then we remain the prediction from the trained model
        if tmp_pred is None:
            tmp_pred = pred


        new_predictions.append(tmp_pred)

    #end_time_data_prediction = time()
    #print("Time usage for dictionary matching: ", (end_time_data_prediction - start_time_data_prediction))

    return new_predictions, percentages

def add_departure_port_code(lst):

    new_df = pd.DataFrame(columns=list(lst[1].keys()))

    if lst[0] == lst[5]:
        cnt = 1
        for i in range(lst[3], lst[3]+lst[-1]):
            for j in range(lst[2].shape[0]):
                if lst[1]['mmsi'][i] == lst[2]['mmsi'][j]:
                    if lst[1]['dest_code'][i] != lst[2]['dest_code'][j]:
                        new_df.loc[cnt] = lst[1].loc[i]
                        new_df.loc[cnt, 'dept_code'] = lst[2].loc[j,'dest_code']
                        cnt+=1
                    else:
                        new_df.loc[cnt] = lst[1].loc[i]
                        new_df.loc[cnt, 'dept_code'] = ""
                        cnt+=1
    else:
        cnt=1
        for i in range(lst[3], lst[3]+lst[4]):
            for j in range(lst[2].shape[0]):
                if lst[1]['mmsi'][i] == lst[2]['mmsi'][j]:
                    if lst[1]['dest_code'][i] != lst[2]['dest_code'][j]:
                        new_df.loc[cnt] = lst[1].loc[i]
                        new_df.loc[cnt, 'dept_code'] = lst[2].loc[j, 'dest_code']
                        cnt += 1
                    else:
                        new_df.loc[cnt] = lst[1].loc[i]
                        new_df.loc[cnt, 'dept_code'] = ""
                        cnt += 1
    new_df = new_df.dropna()
    new_df.to_pickle(CONST_PICKLE_PATH + f"{lst[0]}_final_data.pickle")

def multiprocess_pool(df_final_data, df_last_data):

    ddn = 200
    qtn = df_final_data.shape[0] // ddn
    rdr = df_final_data.shape[0] % ddn
    lsts = []
    for i in range(qtn):
        lsts.append((i + 1, df_final_data.loc[i * ddn: (i + 1) * ddn], df_last_data, i * ddn, ddn, qtn + 1, rdr))
    lsts.append((qtn + 1, df_final_data.loc[qtn * ddn:], df_last_data, qtn * ddn, ddn, qtn + 1, rdr))

    def deleteAllFiles():
        if os.path.exists(CONST_PICKLE_PATH):
            for file in os.scandir(CONST_PICKLE_PATH):
                os.remove(file.path)
            return print("Remove All File")
        else:
            return print("Directory Not Founcd")
    deleteAllFiles()

    pool = Pool(os.cpu_count())
    pool.map(add_departure_port_code, lsts)
    pool.close()
    pool.join()

def concat_pickle():

    file_list = os.listdir(CONST_IN_DEST_PATH)
    if ".DS_Store" in file_list:
        file_list.remove('.DS_Store')
    fileNames = []
    for filename in file_list:
        df = pd.read_pickle(CONST_IN_DEST_PATH + filename)
        fileNames = fileNames + df
        os.remove(CONST_IN_DEST_PATH + filename)

    return fileNames

def deep_learning_data(df_original_data, df_short_dictionary, model):

    if df_original_data is None:
        print("No data has found. Exit immediately")
        exit()

    # 한국선급 전 데이터 저장소
    with open(KR_CONST_PICKLE_FILE, 'rb') as lf:
        dicts = pickle.load(lf)
    lsts = list(dicts.keys())
    # print(lsts)
    # exit()

    start_time_data_filtration = time.time()
    rows_to_be_deleted = []
    real_dest = []
    destinations = []
    mmsi_ship = []
    keys1 = []
    keys2 = []
    for i in range(len(df_original_data["destination"].values)):
        mmsi = df_original_data['mmsi'][i]
        # print(mmsi)
        # exit()
        tmp_dest = df_original_data["destination"][i]
        # print(tmp_dest)
        # exit()
        isValid, cleaned_dest = isValidDest(tmp_dest)
        if isValid == False:
            rows_to_be_deleted.append(i)
        else:
            if cleaned_dest in lsts:
                real_dest.append((mmsi, tmp_dest, cleaned_dest,  dicts[cleaned_dest], 'registered'))
            else:
                mmsi_ship.append(mmsi)
                keys1.append(tmp_dest)
                keys2.append(cleaned_dest)
                destinations.append(cleaned_dest)

    print("Remove rows: ", rows_to_be_deleted)
    data_with_invalid_dest = df_original_data.iloc[rows_to_be_deleted]
    data_with_valid_dest = df_original_data.drop(rows_to_be_deleted, axis=0)
    data_with_valid_dest = data_with_valid_dest.reset_index()

    print("Total destination before filter: ", len(df_original_data))
    print("Total destination after filter: ", len(data_with_valid_dest))
    print("Invalid count number: ", len(data_with_invalid_dest))
    end_time_data_filtration = time.time()

    print("Time usage for data filtration: ", (end_time_data_filtration - start_time_data_filtration))

    # print(real_dest)
    # exit()
    if destinations is not None and len(destinations) > 0:
        # Remove last column because it is just extra information that is not needed
        if "extra" in df_short_dictionary.columns:
            df_short_dictionary = df_short_dictionary.drop("extra", axis=1)

        start_time_data_prediction = time.time()
        #################################
        #   NEURAL NETWORK PREDICTION   #
        #################################
        # destinations = \
        # [ "ITRAN>TRYLA", "ZONA_PESCA>==", "ZONA> DE PESCA",  "NORTHBAY /BARRA", \
        #   "FR MRS > TN LGN", "HELLEVOET<<", "US^0711>0X39", "US^08XG>0902", \
        #   "RUULU>INPPT", "BELEM-PA", "USPAS>USNDX", "ELEFSIS/GR", ">CN NTG", \
        #   ">JP SADO FUTAMI", "PORT HARCOURT/NG PHC", ">JP TYO H", "LKCMB > SGSIN WBGA",\
        #   "SA JED>MY PKL", ">JP IHA", "JP> TRG", "US^0YBG>0XFL"]

        ################################
        predictions, percentages = hybrid_classification(destinations, model, df_short_dictionary, df_short_dictionary)
        # print(len(predictions))
        # exit()
        #predictions, percentages = neural_network_classification(destinations, model, df_short_dictionary)
        for mm, key1, key2, value, percent in zip(mmsi_ship, keys1, keys2,  predictions, percentages):
            real_dest.append((mm, key1, key2, value, percent))


    # 등록된 데이터와 등록되진 않은 데이터 정보
    #########################################################################################################
        tmp = pd.DataFrame(columns=['mmsi', 'input', 'destination', 'port_code', 'percent'])
        idx=0
        for mms, key1, key2, value, percent in real_dest:
            for i in range(df_original_data.shape[0]):
                if df_original_data['mmsi'][i] == mms:
                    df_original_data["dest_code"][i] = value
                    tmp.loc[idx] = [df_original_data['mmsi'][i], df_original_data['destination'][i],\
                                    key2, df_original_data['dest_code'][i], percent]
                    idx+=1
        # tmp.to_csv(CONST_OUT_FILE_PATH + "kr_dest_port_percent_data.csv", index=False, encoding='latin-1')
        # exit()
    #########################################################################################################

        end_time_data_prediction = time.time()
        print("Time usage for data prediction: ", (end_time_data_prediction - start_time_data_prediction))
        print("Finish prediction")

    return df_original_data

# Insert data into the database with execute_values function
def insert_into_table(df, table, template=None):
    """
    Using psycopg2.extras.execute_values() to insert the dataframe
    """
    CONST_CONN = psycopg2.connect(user="user",
                                  password="password",
                                  host="host",
                                  port="port",
                                  database="db")

    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # print(tuples[0])
    # exit()
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    #print(cols)
    #exit()

    # SQL quert to execute
    query = "INSERT INTO %s (%s) VALUES %%s" % (table, cols)
    # print(query)
    # exit()
    cursor = CONST_CONN.cursor()
    try:
        if template is None:
            extras.execute_values(cursor, query, tuples)
        else:
            extras.execute_values(cursor, query, tuples, template=template)
        CONST_CONN.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        CONST_CONN.rollback()
        cursor.close()
        return 1
    print("execute_values() done")
    disconnection(CONST_CONN, cursor)

def main(df_original_data, df_last_data, df_short_dictionary, model):

    df_final_data = deep_learning_data(df_original_data, df_short_dictionary, model)

    geometry = [f"POINT({df_final_data['longitude'].iloc[i]} {df_final_data['latitude'].iloc[i]})" for i in range(len(df_final_data.values))]
    df_final_data["geom"] = geometry

    # Rearrange the dataframe according to the header of the table
    CONST_HEADER_LVI = total_fields_in_postgresQL()
    df_final_data = df_final_data[CONST_HEADER_LVI]
    # print(df_last_data['eta'])
    # exit()

    cnt=0
    for i in range(df_last_data.shape[0]):
        for j in range(df_final_data.shape[0]):
            if df_last_data['imo'][i] == df_final_data['imo'][j]:
                if df_last_data['dest_code'][i] == df_final_data['dest_code'][j]:
                    df_final_data['dest_code'][j] = df_last_data['dest_code'][i]
                    df_final_data['dept_code'][j] = df_last_data['dept_code'][i]
                else:
                    df_final_data['dept_code'][j] = df_last_data['dest_code'][i]
                    #df_final_data['etd'][j] = df_last_data['eta'][i]
                    # print(df_final_data['dept_code'][j])
                    cnt+=1
    print("cnt", cnt)
    # exit()
    print("Finish formation")

    return df_final_data

def kr_data_to_csv_mapping(kr_cur_data):

    gr_ro_bg_cy_mt_df = pd.DataFrame(columns=kr_cur_data.keys())
    au_nz_pg_fj_df = pd.DataFrame(columns=kr_cur_data.keys())
    nl_be_df = pd.DataFrame(columns=kr_cur_data.keys())
    dk_no_se_df = pd.DataFrame(columns=kr_cur_data.keys())
    de_pl_ru_fi_lv_lt_ee_df = pd.DataFrame(columns=kr_cur_data.keys())
    es_pt_gi_df = pd.DataFrame(columns=kr_cur_data.keys())
    in_df = pd.DataFrame(columns=kr_cur_data.keys())
    it_hr_si_df = pd.DataFrame(columns=kr_cur_data.keys())
    gb_ie_is_df = pd.DataFrame(columns=kr_cur_data.keys())
    ca_df = pd.DataFrame(columns=kr_cur_data.keys())
    us_df = pd.DataFrame(columns=kr_cur_data.keys())

    gr_idx, au_idx, nl_idx, dk_idx, de_idx, es_idx, in_idx, \
        it_idx, gb_idx, ca_idx, us_idx = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(kr_cur_data.shape[0]):
        if len(kr_cur_data['dest_code'][i]) == 5:

            if 'GR' in kr_cur_data['dest_code'][i][:2] or 'RO' in kr_cur_data['dest_code'][i][:2] \
                    or 'BG' in kr_cur_data['dest_code'][i][:2] or 'CY' in kr_cur_data['dest_code'][i][:2]\
                    or 'MT' in kr_cur_data['dest_code'][i][:2]:
                gr_ro_bg_cy_mt_df.loc[gr_idx] = kr_cur_data.loc[i]
                gr_idx += 1

            elif 'AU' in kr_cur_data['dest_code'][i][:2] or 'NZ' in kr_cur_data['dest_code'][i][:2] \
                    or 'PG' in kr_cur_data['dest_code'][i][:2] or 'FJ' in kr_cur_data['dest_code'][i][:2]:
                au_nz_pg_fj_df.loc[gr_idx] = kr_cur_data.loc[i]
                au_idx += 1
            elif 'NL' in kr_cur_data['dest_code'][i][:2] or 'BE' in kr_cur_data['dest_code'][i][:2]:
                nl_be_df.loc[gr_idx] = kr_cur_data.loc[i]
                nl_idx += 1
            elif 'DK' in kr_cur_data['dest_code'][i][:2] or 'NO' in kr_cur_data['dest_code'][i][:2] \
                    or 'SE' in kr_cur_data['dest_code'][i][:2]:
                dk_no_se_df.loc[gr_idx] = kr_cur_data.loc[i]
                dk_idx += 1

            elif 'DE' in kr_cur_data['dest_code'][i][:2] or 'PL' in kr_cur_data['dest_code'][i][:2] \
                    or 'RU' in kr_cur_data['dest_code'][i][:2] or 'FI' in kr_cur_data['dest_code'][i][:2]\
                    or 'LV' in kr_cur_data['dest_code'][i][:2] or 'LT' in kr_cur_data['dest_code'][i][:2] \
                    or 'EE' in kr_cur_data['dest_code'][i][:2]:
                de_pl_ru_fi_lv_lt_ee_df.loc[gr_idx] = kr_cur_data.loc[i]
                de_idx += 1

            elif 'ES' in kr_cur_data['dest_code'][i][:2] or 'PT' in kr_cur_data['dest_code'][i][:2] \
                    or 'GI' in kr_cur_data['dest_code'][i][:2]:
                es_pt_gi_df.loc[gr_idx] = kr_cur_data.loc[i]
                es_idx += 1

            elif 'IN' in kr_cur_data['dest_code'][i][:2]:
                in_df.loc[gr_idx] = kr_cur_data.loc[i]
                in_idx += 1

            elif 'IT' in kr_cur_data['dest_code'][i][:2] or 'HR' in kr_cur_data['dest_code'][i][:2] \
                    or 'SI' in kr_cur_data['dest_code'][i][:2]:
                it_hr_si_df.loc[gr_idx] = kr_cur_data.loc[i]
                it_idx += 1

            elif 'GB' in kr_cur_data['dest_code'][i][:2] or 'IE' in kr_cur_data['dest_code'][i][:2] \
                    or 'IS' in kr_cur_data['dest_code'][i][:2]:
                gb_ie_is_df.loc[gr_idx] = kr_cur_data.loc[i]
                gb_idx += 1

            elif 'CA' in kr_cur_data['dest_code'][i][:2]:
                ca_df.loc[gr_idx] = kr_cur_data.loc[i]
                ca_idx += 1

            elif 'US' in kr_cur_data['dest_code'][i][:2]:
                us_df.loc[gr_idx] = kr_cur_data.loc[i]
                us_idx += 1

    gr_ro_bg_cy_mt_df.to_csv(CONST_IN_CSV_PATH + "gr_ro_bg_cy_mt_data.csv", index=False, encoding='utf-8')
    au_nz_pg_fj_df.to_csv(CONST_IN_CSV_PATH + "au_nz_pg_fj_data.csv", index=False, encoding='utf-8')
    nl_be_df.to_csv(CONST_IN_CSV_PATH + "nl_be_data.csv", index=False, encoding='utf-8')
    dk_no_se_df.to_csv(CONST_IN_CSV_PATH + "dk_no_se_data.csv", index=False, encoding='utf-8')
    de_pl_ru_fi_lv_lt_ee_df.to_csv(CONST_IN_CSV_PATH + "de_pl_ru_fi_lv_lt_ee_data.csv", index=False, encoding='utf-8')
    es_pt_gi_df.to_csv(CONST_IN_CSV_PATH + "es_pt_gi_data.csv", index=False, encoding='utf-8')
    in_df.to_csv(CONST_IN_CSV_PATH + "in_data.csv", index=False, encoding='utf-8')
    it_hr_si_df.to_csv(CONST_IN_CSV_PATH + "it_hr_si_data.csv", index=False, encoding='utf-8')
    gb_ie_is_df.to_csv(CONST_IN_CSV_PATH + "gb_ie_is_data.csv", index=False, encoding='utf-8')
    ca_df.to_csv(CONST_IN_CSV_PATH + "ca_data.csv", index=False, encoding='utf-8')
    us_df.to_csv(CONST_IN_CSV_PATH + "us_data.csv", index=False, encoding='utf-8')

def kr_data_to_zip_making():

    zip_file = zipfile.ZipFile(CONST_IN_ZIP_PATH + "kr_total_data.zip", "w")  # "w": write 모드
    for file in os.listdir(CONST_IN_CSV_PATH):
        if file.endswith('.csv'):
            zip_file.write(os.path.join(CONST_IN_CSV_PATH, file), compress_type=zipfile.ZIP_DEFLATED)

    zip_file.close()

def get_dataFrame_and_Table():

    whatDate = datetime.now()
    currentDay = whatDate.strftime('%Y%m')
    CONST_CUR_TABLE = 'lvi_prm_' + currentDay + '%'

    connection, cursor, table_name= connect_DB(CONST_CUR_TABLE, CONST_KR_TABLE)
    disconnection(connection, cursor)
    # connection, cursor, kr_df = connect_DB4(CONST_KR_TABLE)
    # disconnection(connection, cursor)
    # connection, cursor, lvi_df = connect_DB4(table_name)
    # disconnection(connection, cursor)
    print("table name: ", table_name)

    return table_name

def pickle_update_data(items):

    lsts = []
    items[2].reset_index(drop=True, inplace=True)
    items[1].reset_index(drop=True, inplace=True)
    # print(items[2])
    # print(items[2].shape[0])
    # exit()

    for i in range(items[2].shape[0]):
        if items[2]['imo'][i] is None:
            continue
        for j in range(items[1].shape[0]):
            if items[1]['imo'][j] is None:
                continue
            if int(items[1]['imo'][j]) == int(items[2]['imo'][i]):
                imo = items[1]['imo'][j]
                a1 = str(items[1]['dest_code'][j])
                a2 = str(items[1]['dept_code'][j])
                if a1 == None:
                    a1 = ''
                if a2 == None:
                    a2 = ''
                lsts.append((imo, a1, a2))
        if i == items[2].shape[0]-1:
            print("Sleep 5 seconds from now on>>>")
            time.sleep(5)
            print("wake up!!!!")

    if len(lsts)>0:
        with open(CONST_IN_DEST_PATH + f"{items[0]}_dest_dept.pickle", 'wb') as lf:
            pickle.dump(lsts, lf)

def put_lvi_prm_table_name():

    conn = psycopg2.connect(user="user",
                            password="password",
                            host="host,
                            port="port",
                            import os
import pandas as pd
import sqlalchemy
from tqdm import tqdm
from sqlalchemy.sql import text
from multiprocessing import Pool



HOST = "host"
DATABASE= "database"
USERNAME = "username"
PASSWORD = "password"
PORT = port
SCHEMA = 'public'
TABLENAME = "lvi_prm_2022"

CSVNAME = "220707-56척+14척 선박리스트.xlsx"

url = f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'
engine = sqlalchemy.create_engine(url)



sql = (
        f"""
            SELECT tablename FROM pg_tables
            WHERE tablename LIKE '{TABLENAME}%'
    
        """)

with engine.connect().execution_options(autocommit=True) as conn:
    query = conn.execute(text(sql))

total_tables = query.fetchall()

def extract_data(mmsi):

    lsts = []
    for i in tqdm(range(len(total_tables))):
        sql = (
            f"""
                    SELECT * FROM {total_tables[i][0]}
                    WHERE mmsi = {mmsi};

                """)
        with engine.connect().execution_options(autocommit=True) as conn:
            query = conn.execute(text(sql))

        df = pd.DataFrame(query.fetchall())
        try:
            df = df.drop(['geom'], axis=1)
        except:
            print("mmsi: ", mmsi)
        lsts.append(df)

    total_df = pd.concat(lsts)
    total_df.to_csv(f"56_data/56_mmsi_ship_data_{mmsi}.csv", index=False, sep=',', encoding='utf-8')

if __name__ == "__main__":

    df_mmsi = pd.read_excel(CSVNAME, sheet_name='Sheet1')
    df_mmsi = df_mmsi.loc[2:57, 'Unnamed: 2']
    lsts_mmsi = list(df_mmsi)

    cnt = os.cpu_count()
    pool = Pool(cnt)
    pool.map(extract_data, lsts_mmsi)
    pool.close()
    pool.join()database="oceanlook")
    today = datetime.now()
    lvi_postfix1 = today.strftime("%Y%m%d%H")
    intmm = int(today.strftime("%M"))
    stymm = ""

    if 0 <= intmm < 30:
        stymm = "10";
    elif intmm >= 30 and intmm <= 60:
        stymm = "40";

    lvi_postfix = lvi_postfix1 + "" + stymm + "00"

    list_table_name = "lvi_prm_tables"
    CONST_TB = "lvi_prm_" + lvi_postfix

    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO " + list_table_name + " (tb_name, tb_usage, tb_time, tb_endtime) VALUES ( '" + CONST_TB + "', '1', now(), now())")
    conn.commit()
    cursor.close()
    conn.close()



if __name__ == "__main__":

    MODEL = load_model(CONST_SAVED_MODEL)
    lists = []
    dept_dest_code = ["NL TNZ", "GB MID-L"]
    for i in range(len(dept_dest_code)):
        tmp_dest = dept_dest_code[i]
        isValid, cleaned_dest = isValidDest(tmp_dest)
        lists.append(cleaned_dest)

    predictions, percentages = hybrid_classification(lists, MODEL, DF_SHORT_DICTIONARY, DF_SHORT_DICTIONARY)
    print(predictions)
    exit()
    # dest = ["NL TNZ", "GB MID-L"]
    # a, b = hybrid_classification(dest, MODEL, DF_SHORT_DICTIONARY, DF_SHORT_DICTIONARY)


    start = time.time()
    # tableName
    whatDate = datetime.now()
    yearMonthDay = whatDate.strftime('%Y%m')
    CONST_CUR_TABLE = 'lvi_prm_' + yearMonthDay + '%'
    # print(CONST_CUR_TABLE)
    # exit()
    CONST_KR_MAPING_TABLE = "test_lvi_prm"

    # ship_imo = tuple(DF_KR_SHIP_DICTIONARY['IMO No,'])
    COL = "imo"
    cur_records, past_records = extract_data_from_postgresQL(CONST_CUR_TABLE, CONST_KR_MAPING_TABLE, COL)
    # exit()
    # LSTM Model 로드

    MODEL = load_model(CONST_SAVED_MODEL)
    df_final_data = main(cur_records, past_records, DF_SHORT_DICTIONARY, MODEL)
    # print(list(df_final_data['etd']))
    # exit()

    insert_into_table(df_final_data, CONST_KR_TABLE)



    table_name = get_dataFrame_and_Table()
    update_table_in_postgresQL(CONST_KR_TABLE, table_name)
    put_lvi_prm_table_name()



    end = time.time()
    print("elapsed time: {}".format(end-start))


    #connection, cursor, kr_cur_data = connect_DB4(CONST_KR_TABLE)
    #disconnection(connection, cursor)
    #kr_data_to_csv_mapping(kr_cur_data)
    #kr_data_to_zip_making()
    """
    # CSV FILE
    AU_FILE = [CONST_IN_CSV_PATH + "au_nz_pg_fj_data.csv"]
    CA_FILE = [CONST_IN_CSV_PATH + "ca_data.csv"]
    DE_FILE = [CONST_IN_CSV_PATH + "de_pl_ru_fi_lv_lt_ee_data.csv"]
    DK_FILE = [CONST_IN_CSV_PATH + "dk_no_se_data.csv"]
    ES_FILE = [CONST_IN_CSV_PATH + "es_pt_gi_data.csv"]
    GB_FILE = [CONST_IN_CSV_PATH + "gb_ie_is_data.csv"]
    GR_FILE = [CONST_IN_CSV_PATH + "gr_ro_bg_cy_mt_data.csv"]
    IN_FILE = [CONST_IN_CSV_PATH + "in_data.csv"]
    IT_FILE = [CONST_IN_CSV_PATH + "it_hr_si_data.csv"]
    NL_FILE = [CONST_IN_CSV_PATH + "nl_be_data.csv"]
    US_FILE = [CONST_IN_CSV_PATH + "us_data.csv"]
    ZIP_FILE = [CONST_IN_ZIP_PATH + "kr_total_data.zip"]

    file_lists = [AU_FILE, CA_FILE, DE_FILE, DE_FILE, ES_FILE, GB_FILE, GR_FILE, \
                  IN_FILE, IT_FILE, NL_FILE, US_FILE, ZIP_FILE]

    for fileList in file_lists:
        send_mail_to_client(FROM, TO, OUTLOOKMAIL, PASSWARD, fileList)
    """


