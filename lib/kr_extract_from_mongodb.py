import pandas as pd
import numpy as np
import os
import inspect
from time import time
from datetime import datetime
from pymongo import MongoClient
from pyapi.lib.preprocessing_road_dictionary import *

# path="/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyAPI/in/"
# csv_name="kr_ship_1877.csv"
# # vessel imo that extracts from Hundai_Global
# vessel_info = pd.read_csv(path + csv_name, sep=',', encoding='utf-8')
# #vessel = vessel_info.fillna(0)
# ship_num_lists = vessel_info['IMO No,']
# ship_num_lists = sorted(set(ship_num_lists))
# ship_num_lists = ship_num_lists[100:101]
# print(ship_num_lists)


# mongodb 연동
_id = "ecomarine"
password = "Ecomarine1!"
host = "43.200.0.13"
port = 27710
CONST_MONGO_LOCAL_DB = "ecomarine"
CONST_MONGO_LOCAL_URL = f"mongodb://{_id}:{password}@{host}:{port}/{CONST_MONGO_LOCAL_DB}?authSource=admin"
CONST_MONGO_LOCAL_COLLECTION = "vesselNaviData_Spire_"


def aggregate_mongo_data_from(collection_name, pipline={}):

    return collection_name.aggregate(pipline, allowDiskUse=True)


def get_mongodb_pipeline(mmsi, date_start=datetime.now(), date_end=datetime.now()):
    search_query = {}
    if mmsi:
        search_query["mmsi"] = mmsi
    search_query["positionUpdated"] = {}
    search_query["positionUpdated"]["$gte"] = str(date_start)
    search_query["positionUpdated"]["$lte"] = str(date_end)


    pipeline_match = {
        "$match": {
            "$and": [search_query]
        }
    }

    pipeline_sort = {
        "$sort": {
            "imoNumber": 1,
            "positionUpdated": 1
        }
    }

    pipeline_project = {
        "$project": {
            "_id": 0,
            "positionUpdated": 1,
            "mmsi":1,
            "imoNumber":1,
            "longitude": 1,
            "latitude": 1,
            "sog":1,
            "cog":1,
            "navStatus":1,
            "destination":1
        }
    }

    pipeline_unset = {
        "$unset": ["_id"]
    }

    return [pipeline_match, pipeline_project, pipeline_unset, pipeline_sort]


def months_between(datefrom, dateto):
    cur_date = datetime.strptime(datefrom, '%Y-%m-%d')
    end_date = datetime.strptime(dateto, '%Y-%m-%d')

    cur_year = cur_date.year
    cur_month = cur_date.month

    end_year = end_date.year
    end_month = end_date.month

    date_postfix_list = []
    while (cur_year, cur_month) <= (end_year, end_month):
        date_postfix_list.append(f"""{cur_year}{cur_month if cur_month >= 10 else f"0{cur_month}"}""")
        if cur_month == 12:
            cur_month = 1
            cur_year += 1
        else:
            cur_month += 1

    return date_postfix_list


def mongodb_conn(url, dbname):
    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(url)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client[dbname]


def get_history_data(mmsi=None, datefrom=None, dateto=None):
    today_date = datetime.now()
    date_start = today_date.strftime("%Y-%m-%d")
    date_end = today_date.strftime("%Y-%m-%d")

    date_postfix_list = months_between(datefrom, dateto)


    if datefrom is not None:
        date_start = datetime.strptime(datefrom, '%Y-%m-%d')

    if dateto is not None:
        date_end = datetime.strptime(dateto, '%Y-%m-%d')

    dbname = mongodb_conn(CONST_MONGO_LOCAL_URL, CONST_MONGO_LOCAL_DB)
    collection_lists = dbname.list_collection_names()

    pipeline = get_mongodb_pipeline(mmsi, date_start, date_end)

    accepted_data = []
    for date_postfix in date_postfix_list:
        collection_name = CONST_MONGO_LOCAL_COLLECTION + date_postfix

        if collection_name not in collection_lists:
            print(collection_name, "is not found in the database, skip it")
            continue

        mg_col_conn = dbname[collection_name]
        print("COLLECTION_NAME: ", collection_name)

        print("pipeline", pipeline)

        start_time = time()
        data = aggregate_mongo_data_from(mg_col_conn, pipeline)

        end_time = time()
        print("Time usage for mongodb aggregation:", (end_time - start_time))

        start_time = time()
        for datum in data:
            datum['imo'] = datum['imoNumber']
            del datum['imoNumber']
            datum['nav_status'] = datum['navStatus']
            del datum['navStatus']
            datum['dt_pos_utc'] = datum['positionUpdated']
            del datum['positionUpdated']
            accepted_data.append(datum)
        end_time = time()
        print("Time usage for turning mongodb to python:", (end_time - start_time))
        print(len(accepted_data))
        if len(accepted_data) == 0:
            continue
        print(accepted_data[1])
    accepted_data = pd.DataFrame(accepted_data)

    if accepted_data.empty:
        print("Data is empty!")
    else:
        # print("&&&&&&&")
        # exit()
        accepted_data = accepted_data.astype({"mmsi": int,
                                              "imo":int,
                                              'longitude': float,
                                              'latitude': float,
                                              'sog': float,
                                              'cog':float,
                                              "nav_status":str
                                              })

    return accepted_data  # return data with 200 OK


def main(records):
    important_cols = ['mmsi','imo','longitude', 'latitude', \
                      'sog', 'cog', 'destination', 'nav_status', 'dt_pos_utc']
    show_number_of_missing_values_in(records, important_cols)

    # path_dir = "C:/Users/SAVANNA/Dropbox/HANEUL/AIS/data2"  # *
    # with open(f'{path_dir}/records.pickle', 'wb') as f: # *
    #    pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)

    # with open(f'{path_dir}/records.pickle', 'rb') as f:
    #    records = pickle.load(f)

    # 1. Delete records which have not cetain keys that corresponds to the option
    print('\n1. Delete records which have not cetain keys that corresponds to the option')
    option = 'imo'  # 'imo', 'mmsi', or 'all' *
    delete_non_primary_key_records(records, option=option)  #

    # 2. Find and remove key values which is one-to-many relation with another key and corresponds to the option
    # option = 'all' 선택시 서로 다른 ID와 다중 연결 된 ID(mmsi, imo)를 가지는 레코드 모두 삭제
    # option = 'imo' 선택시 여러 mmsi와 다중 연결된 imo를 가지는 레코드 모두 삭제
    # option = 'mmsi' 선택시 여러 imo와 다중 연결된 mmsi를 가지는 레코드 모두 삭제
    print(
        '\n2. Find and remove key values which is one-to-many relation with another key and corresponds to the option')
    option = 'mmsi'  # 'imo', 'mmsi', or 'all' *
    delete_multiple_linked_primary_key(records, option=option)

    # (option) Convert time series sailing records to ship-specific sailing records
    records = group_raw_data_by_ship_id(records)

    # (option) Check all navigation status
    # print(get_all_navigation_status(records))

    # 3. Sort by time series
    print('\n3. Sort by time series')
    sort_by_time_series(records)
    # print(records[0].info())
    # exit()
    # 4. Interpolate 'sog'
    print('\n4. Interpolation "sog"')
    for i in tqdm(range(len(records))):
        records[i] = interpolate_by_mean(records[i])

    # 6. Slicing data based on a single voyage
    print('\n5. Slicing data based on a single voyage')
    # Please check func : get_slicing_index's description to set up arguments
    # crit = ['sog', 'destination', 'port', 'status']
    pre_crit = ['sog']  # *
    post_crit = ['sog', 'destination', 'port', 'status']  # *
    base_score = 2  # *
    semi_diameter = 50  # unit = km, setting range of adjacent areas *
    crit_sog = 3  # *

    sailing_lst = []
    for j in tqdm(range(len(records))):
        records[j] = records[j].reset_index(drop=True)
        slicing_index = get_slicing_index(records[j], pre_crit, post_crit, base_score, semi_diameter, crit_sog)
        for i in list(range(len(slicing_index)))[:-1]:
            sailing_lst.append(records[j].loc[slicing_index[i]: slicing_index[i + 1]])

    # 7. Remove receiving error data point(AIS)
    print('\n6. Remove receiving error data point(AIS)')
    max_sog = 40  # *

    for i in tqdm(range(len(sailing_lst))):
        sailing_lst[i] = sailing_lst[i].reset_index(drop=True)
        # print(sailing_lst[i])
        # exit()
        sailing_lst[i] = delete_recieving_error_data(sailing_lst[i], max_sog)
    # print(sailing_lst)
    # exit()
    # 8. Remove data with small records or total sailing distance
    print('\n7. Remove data with small records or total sailing distance')
    min_dist = 50  # minimum total sailing distance, unit = km *
    min_len = 20  # minimum length of record *

    drop_lst = []
    for idx in tqdm(range(len(sailing_lst))):
        if is_short_data(sailing_lst[idx], min_dist, min_len):
            drop_lst.append(idx)
    drop_lst.reverse()
    for idx in drop_lst:
        sailing_lst.pop(idx)

    # 9. Get origin, destination of sailing route
    print('\n8. Get origin, destination of sailing route')
    sailing_records = []
    for i in tqdm(range(len(sailing_lst))):
        sailing_lst[i] = sailing_lst[i].reset_index(drop=True)
        record = get_o_d_record(sailing_lst[i], semi_diameter, port_list)
        if record: sailing_records.append(record)

    return sailing_records


if __name__ == "__main__":

    # x = datetime.utcfromtimestamp(1660760552)
    # print(x)
    # exit()

    mmsi = str(373595000)
    datefrom = "2022-09-01"
    dateto = "2022-09-06"

    records = [get_history_data(mmsi, datefrom, dateto)]


    dicts = {}
    if records[0].shape[0]:
        sailing_records = main(records)
    # print(">>>>", sailing_records[0])
    # exit()
    for i in range(len(sailing_records)):
        dicts[i] = \
            (sailing_records[i]['origin']['port_code'], sailing_records[i]['destination']['port_code'])
    print(dicts)







