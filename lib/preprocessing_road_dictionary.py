"""
Preprocessing and Clustering Analysis about Sailing Data

Author : HANEUL KIM
Email : rgs6827@korea.ac.kr
Python Version : 3.8.8
Package Version :
    Package Name         Version
    ------------         -------
    tqdm                 4.62.0
    pandas	             1.3.1
    numpy	             1.20.3
    haversine	         2.3.1
    rdp	                 0.8
    sklearn	             0.0
    scipy	             1.6.2
    geopandas	         0.9.0
    shapely	             1.7.1
"""
import sys
sys.path.append('../')
import os
import pickle
import gc
import pytz

from tqdm import tqdm
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

from haversine import haversine, haversine_vector
from rdp import rdp
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster, linkage

import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.geometry import MultiLineString, MultiPolygon
from pyapi.apiClass.dept_dest_code_api import CONST_IN_FILE_PATH

# from global_land_mask import globe


# NOTICE : recommended to store intermediate data.

# 0-0. Load port list
print("0-0. Load port_list")
# path_dir = "C:/Users/SAVANNA/Dropbox/HANEUL/AIS/data"  # *

file_name = "final_port_code_dictionary.csv"
port_list = pd.read_csv(CONST_IN_FILE_PATH + file_name, encoding='cp949')
port_list = port_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

port_list = port_list[port_list['lat'].notnull()]  # *
port_list = port_list[port_list['lon'].notnull()]  # *
port_list['lat'] = port_list['lat'].astype(np.float64)  # *
port_list['lon'] = port_list['lon'].astype(np.float64)  # *
filename = "sailing_o_d_record"  #########################################################################################################################################################################
sailing_files = CONST_IN_FILE_PATH + filename + ".csv"

# 5. Get coordinate ranges for finding neighbor ports
print('\n0-1. Get coordinate ranges for finding neighbor ports')
len_port = len(port_list)
lon_lst_by_lat = [111.322, 110.902, 109.643, 117.553, 114.650, 100.953, \
                  96.490, 91.290, 85.397, 78.850, 71.700, 63.997, 55.803, \
                  47.178, 38.188, 28.904, 19.394, 9.735]
lat_lst = [i * 5 for i in range(18)]
lon_1degree_dist = {lat_lst[i]: lon_lst_by_lat[i] for i in range(len(lat_lst))}
lat_1degree_dist = 111

# 0. Data load
# print('\n0-3. Data load')
# # 0-1. Load sailing records
# # path_dir = "C:/Users/SAVANNA/Dropbox/HANEUL/AIS/data2/ship" # *
# inPath = "/Users/jeongtaegun/Desktop/surver_project/trackAndODpairs/pyProject/csv/"
# directories = os.listdir(inPath)
# drts = []
# for drt in directories:
#     if drt == ".DS_Store":
#         continue
#     drts.append(f'{inPath}{drt}')
#
# drts = sorted(drts)

def load_raw_data(drts):
    r"""
    Method for loading data

    Parameters
    ----------
    path_dir : string - raw data's directory path

    Returns
    -------
    records : list - navigation history list
        Each element in the list is a record of a single ship's voyage.

    """
    records = []
    for drt in tqdm(drts):
        files = os.listdir(drt)
        for file in files:
            df = pd.read_csv(drt + '/' + file, encoding='latin-1', low_memory=False)
            if not df.empty:
                df.rename(columns={'nav_status_code':'nav_status'}, inplace=True)
                del df['_id']
                for i in range(df.shape[0]):
                    if df.loc[i, 'nav_status'] == 0:
                        df.loc[i, 'nav_status'] ='Under way using engine'
                    elif df.loc[i, 'nav_status'] == 1:
                        df.loc[i, 'nav_status'] ='At anchor'
                    elif df.loc[i, 'nav_status'] == 2:
                        df.loc[i, 'nav_status'] ='Not under command'
                    elif df.loc[i, 'nav_status'] == 3:
                        df.loc[i, 'nav_status'] ='Restricted manoeuverability'
                    elif df.loc[i, 'nav_status'] == 4:
                        df.loc[i, 'nav_status'] ='Constrained by her draught'
                    elif df.loc[i, 'nav_status'] == 5:
                        df.loc[i, 'nav_status'] ='Moored'
                    elif df.loc[i, 'nav_status'] == 6:
                        df.loc[i, 'nav_status'] ='Aground'
                    elif df.loc[i, 'nav_status'] == 7:
                        df.loc[i, 'nav_status'] ='Engaged in Fishing'
                    elif df.loc[i, 'nav_status'] == 8:
                        df.loc[i, 'nav_status'] ='Under way sailing'
                    elif df.loc[i, 'nav_status'] == 9:
                        df.loc[i, 'nav_status'] ='Reserved for future amendment of Navigational Status for HSC'
                    elif df.loc[i, 'nav_status'] == 10:
                        df.loc[i, 'nav_status'] ='Reserved for future amendment of Navigational Status for WIG'
                    elif df.loc[i, 'nav_status'] == 11:
                        df.loc[i, 'nav_status'] ='Reserved for future use'
                    elif df.loc[i, 'nav_status'] == 12:
                        df.loc[i, 'nav_status'] ='Reserved for future use'
                    elif df.loc[i, 'nav_status'] == 13:
                        df.loc[i, 'nav_status'] ='Reserved for future use'
                    elif df.loc[i, 'nav_status'] == 14:
                        df.loc[i, 'nav_status'] ='AIS-SART is active'
                    elif df.loc[i, 'nav_status'] == 15:
                        df.loc[i, 'nav_status'] ='Not defined (default)'
                records.append(df)
    return records

def get_all_navigation_status(records):
    r"""
    Obtain navigation status observed in all navigation records.

    Parameters
    ----------
    records : list - navigation history list

    Returns
    -------
    status : set - all navigation status

    """
    status = set()
    for i in range(len(records)):
        status = status.union(set(records[i].nav_status.unique()))
    return status

def delete_non_primary_key_records(records, option='imo'):
    r"""
    Find and remove records which does not have keys corresponds to the option
    if option = 'imo' than find a record which has a null value with a IMO value.

    Parameters
    ----------
    records : list - list of dataframe which is a sailing record
    option : str - 'imo', 'mmsi',or 'all'

    """
    opt = {'imo': ['imo'], 'mmsi': ['mmsi'], 'all': ['mmsi', 'imo']}

    for i, record in tqdm(enumerate(records)):
        crit = record.imo.notnull() | ~record.imo.notnull()
        for keys in opt[option]:
            crit = crit & record[keys].notnull()
        if not crit.all():
            records[i] = record[crit]

def delete_multiple_linked_primary_key(records, option='mmsi'):
    r"""
    Find and remove key values which is one-to-many relation with another key and corresponds to the option
    if option = 'imo' than find IMO values which is one-to-many relation with 'MMSI' and remove them

    Parameters
    ----------
    records : list - list of dataframe which is a sailing record
    option : str - 'imo', 'mmsi', 'all'

    """

    opt = {'imo': [('imo', 'mmsi')], 'mmsi': [('mmsi', 'imo')], 'all': [('mmsi', 'imo'), ('imo', 'mmsi')]}

    mmsi_set = []
    imo_set = []

    for record in tqdm(records):
        mmsi_set.extend(set(record.mmsi))
        imo_set.extend(set(record.imo))
    mmsi_set = set(mmsi_set)
    imo_set = set(imo_set)
    print(f'Number of MMSI in INPUT : {len(mmsi_set)}')
    print(f'Number of IMO in INPUT : {len(imo_set)}')

    ids_dic = {'mmsi': defaultdict(lambda: 0), 'imo': defaultdict(lambda: 0)}

    errors = {'mmsi': [], 'imo': []}
    for i in tqdm(range(len(records))):
        for idx in range(len(records[i])):
            for keys in opt[option]:
                if ids_dic[keys[0]][records[i][keys[0]].iloc[idx]] == 0:
                    ids_dic[keys[0]][records[i][keys[0]].iloc[idx]] = records[i][keys[1]].iloc[idx]
                else:
                    if ids_dic[keys[0]][records[i][keys[0]].iloc[idx]] != records[i][keys[1]].iloc[idx]:
                        errors[keys[0]].append(records[i][keys[0]].iloc[idx])

    for keys in opt[option]:
        errors[keys[0]] = list(set(errors[keys[0]]))

    mask = [None for i in range(len(records))]
    for i in tqdm(range(len(records))):
        mask[i] = records[i][opt[option][0][0]].isin(errors[opt[option][0][0]]).to_numpy()
        if option == 'all':
                mask[i] = mask[i] + records[i][opt[option][1][0]].isin(errors[opt[option][1][0]]).to_numpy()
        records[i] = records[i][list(~mask[i])]

#해당 코드는 imo를 기준으로 그룹핑하는 함수
def group_raw_data_by_ship_id(records):
    r"""
    transform dataframe which is a sailing record by time series
    into dataframe which is a sailing record by a ship

    Parameters
    ----------
    records : list - list of dataframe which is a sailing record

    Returns
    -------
    records : list - list of dataframe which is a sailing record

    """
    imo_set = []
    for record in tqdm(records):
        imo_set.extend(set(record.imo))
    imo_set = list(set(imo_set))
    print(f'Number of IMO in INPUT : {len(imo_set)}')

    records_dic = {}
    for imo in imo_set:
        records_dic[imo] = pd.DataFrame(None, columns=records[0].columns)

    lst = []
    for i in tqdm(range(len(records))):
        if i % 100 == 0:
            lst.append(records[i])
        else:
            lst[int(i / 100)] = lst[int(i / 100)].append(records[i])

    for record in tqdm(lst):
        for imo in imo_set:
            records_dic[imo] = records_dic[imo].append(record[record['imo'] == imo])

    records = list(records_dic.values())
    return records


def sort_by_time_series(records):
    r"""
    Sort each voyage record in time series order

    Parameters
    ----------
    records : list - navigation history list

    """
    for i in tqdm(range(len(records))):
        for j in range(len(records[i])):
            records[i]['dt_pos_utc'][j] = records[i]['dt_pos_utc'][j][:20]

        records[i]['dt_pos_utc'] = pd.to_datetime(records[i]['dt_pos_utc'], infer_datetime_format=True)
        records[i].reset_index(drop=False, inplace=True)
        records[i] = records[i].sort_values('dt_pos_utc')


def check_missing_values_in(records, column_name):
    r"""
    Calculate a number of missing values in a particular column of records.

    Parameters
    ----------
    records : list - navigation history list
    column_name : list - list of column names want to check.

    Returns
    -------
    null_data_idx : list
        i th list element is a number of i th record's missing values

    """
    null_data_idx = []
    for i in range(len(records)):
        if sum(records[i][column_name].isnull()): null_data_idx.append(i)
    return null_data_idx


def show_number_of_missing_values_in(records, important_cols):
    r"""
    Show index of record which has missing values in important variables

    Parameters
    ----------
    records : list - navigation history list
    important_cols : list - list of important variables

    """
    for column_name in important_cols:
        print()
        print(f"=={column_name}==")
        print(check_missing_values_in(records, column_name))


def get_slicing_data(records):
    r"""
    Use this function to distinguish a single voyage.

    Parameters
    ----------
    records : list - navigation history list

    Returns
    -------
    sailing_list : list
        i th list element is a single voyage record

    """
    sailing_lst = []
    slicing_criteria = ['At Anchor', 'Moored']
    for cnt, record in tqdm(enumerate(records)):
        start_idx = None
        current_nav_status = record.nav_status.iloc[0]
        for i in tqdm(range(1, len(record), 1)):
            before_nav_status = current_nav_status
            current_nav_status = record.iloc[i].nav_status
            if current_nav_status not in slicing_criteria \
                    and before_nav_status in slicing_criteria:
                start_idx = i

            elif current_nav_status in slicing_criteria \
                    and before_nav_status not in slicing_criteria:
                end_idx = i

                if start_idx:
                    sailing_lst.append(record.iloc[start_idx:end_idx])
                    start_idx = None
    return sailing_lst


def get_distance_sailed(point_array):
    r"""
    Calculate total distance of a route

    Parameters
    ----------
    point_array : numpy.array - ( latitude, longitude ) of a sailing record

    Returns
    -------
    dist : float - total distance of a route ( unit : km )

    """
    dist = 0.0
    for i in [i for i in range(1, len(point_array), 1)]:
        dist = dist + haversine(point_array[i - 1], point_array[i])
    return dist


def is_short_data(record, min_dist=None, min_len=None):
    r"""
    Exclude if the shortest distance between the start and end points of
    a extracted single voyage is less than min_dist or
    data record's length is less than min_len.

    Parameters
    ----------
    record : a single voyage record
    min_len : int - number of rows
    min_dist : int or float - exclude criteria for a single voyage ( unit : km )

    Returns
    -------
    boolean - if record is short data then returns true

    """
    if min_len:
        if len(record) < min_len:
            return True
    if min_dist:
        if haversine(record.iloc[0][['latitude', 'longitude']].to_numpy(), \
                     record.iloc[-1][['latitude', 'longitude']].to_numpy()) < min_dist:
            return True
    return False


def delete_recieving_error_data(record, max_sog):
    r"""
    identify records that are out of feasible travel distance

    Parameters
    ----------
    record : list - a single voyage
    max_sog : int or float - maximum sog of ship ( unit : km )

    Returns
    -------
    drop_idx_lst : list - list of row's index which has a coordinate difficult to reach in time

    """
    record = record.astype({'latitude':'float', 'longitude':'float'})
    record.info()
    bp2cp_dist = haversine_vector(
        record.loc[:max(record.index)-1, ['longitude', 'latitude']],
        record.loc[1:, ['longitude', 'latitude']],
        unit='km')
    bp2cp_time = (record.loc[1:,'dt_pos_utc'].to_numpy()\
                  - record.loc[:max(record.index)-1, 'dt_pos_utc'].to_numpy()).astype('timedelta64[s]')
    bp2cp_time = bp2cp_time.astype('float')
    possible_dist = bp2cp_time * ((max_sog * 1.852)/(60*60))
    idx = np.where((possible_dist - bp2cp_dist) < 0 )[0] + 1
    record = record.drop(idx)
    record.reset_index(drop = True, inplace = True)
    return record

def has_long_time_interval(sailing, max_time):
    r"""
    Verify that a long time interval exists between continuous sailing position receiving data

    Parameters
    ----------
    sailing_records : dataframe - a single voyage record
    max_time : float or int - time interval allowed

    Returns
    -------
    boolean - if a record has a long time interval then returns true

    """
    max_time = max_time * 60 * 60
    base_point_idx = sailing.index[0]
    for current_point_idx in sailing.index[1:]:
        bp2cp_time = (
                    sailing.loc[current_point_idx].dt_pos_utc - sailing.loc[base_point_idx].dt_pos_utc).total_seconds()
        if bp2cp_time >= max_time:
            return True
    return False

def get_lon_1degree_dist(left, right, lat):
    r"""
    Binary search function to obtain approximate distance of 1 degree longitude
    from a specific latitude

    Parameters
    ----------
    left : int - left index of search area
    right : int - right index of search area
    lat : int or float - latitude

    Returns
    -------
    float - approximate distance of 1 degree longitude

    """
    if left > right:
        if left == len(lat_lst):
            left = len(lat_lst) - 1
        if lat - lat_lst[right] > lat_lst[left] - lat:
            return lon_1degree_dist[lat_lst[right]]
        else:
            return lon_1degree_dist[lat_lst[left]]
    middle = (left + right) // 2
    if lat_lst[middle] == lat:
        return lon_1degree_dist[lat_lst[middle]]
    if lat_lst[middle] > lat:
        return get_lon_1degree_dist(left, middle - 1, lat)
    else:
        return get_lon_1degree_dist(middle + 1, right, lat)

def get_search_range(point, semi_diameter):
    r"""
    Get search range to find nearby ports at a point

    Parameters
    ----------
    point : tuple - ( latitude, longitude )

    Returns
    -------
    dictionary - start and end point of latitude and longitude

    """
    vertical_interval = semi_diameter / lat_1degree_dist
    horizontal_interval = semi_diameter / get_lon_1degree_dist(0, len(lat_lst) - 1, abs(point[0]))

    lat_start = point[0] - vertical_interval
    lat_end = point[0] + vertical_interval
    lon_start = point[1] - horizontal_interval
    lon_end = point[1] + horizontal_interval

    # 위경도 (-90,90), (-180,180) 을 벗어난 경우
    if lat_start < -90: lat_start = -90
    if lat_end > 90: lat_end = 90
    if lon_start < -180: lon_start = 360 + lon_start
    if lon_end > 180: lon_end = lon_end - 360

    return {'lat': (lat_start, lat_end), 'lon': (lon_start, lon_end)}

def interpolate_by_mean(record):
    r"""
    Interpolates the missing values in the column 'sog' as the mean of its neighbors

    Parameters
    ----------
    record : dataframe - a record of a ship's voyage during the data collection period

    Returns
    -------
    record : dataframe

    """
    col_name = 'sog'
    record.reset_index(drop=True, inplace=True)

    null_lst = record[col_name].isnull()
    max_idx = max(null_lst.index)
    if not null_lst[:][null_lst[:] == True].empty:
        idx = min(null_lst[:][null_lst[:] == True].index)
    else:
        return record
    if null_lst[idx]:
        pro_idx = idx
        while null_lst[pro_idx]:
            if pro_idx != max_idx:
                pro_idx = pro_idx + 1
            else:
                raise Exception(f"None of record has {col_name} data")
                break
    record.loc[idx : pro_idx - 1, col_name] = record.loc[pro_idx, col_name]
    null_lst[idx : pro_idx] = False
    if not null_lst[pro_idx:][null_lst[pro_idx:] == True].empty:
        idx = min(null_lst[pro_idx:][null_lst[pro_idx:] == True].index)
    else:
        return record
    value = None
    while idx <= max_idx:
        if null_lst[idx]:
            pro_idx = idx
            while null_lst[pro_idx]:
                if pro_idx != max_idx:
                    pro_idx = pro_idx + 1
                else:
                    value = record.loc[idx - 1, col_name]
                    break
            if value == None:
                value = (record.loc[pro_idx, col_name] + record.loc[idx-1, col_name])/2
            record.loc[idx: pro_idx - 1, col_name] = value
            null_lst[idx: pro_idx] = False
            if not null_lst[pro_idx:][null_lst[pro_idx:] == True].empty:
                idx = min(null_lst[pro_idx:][null_lst[pro_idx:] == True].index)
            else:
                break
            value = None
    return record

def get_nearest_port_idx(point, port_list):
    r"""
    Returns the index of a port which is nearest to the point

    Parameters
    ----------
    point : list - [ latitude, longitude ]
    port_list : dataframe - ports information data

    Returns
    -------
    nearest : int, float - the nearest port index, a distance from the point to the port

    """
    nearest = [None, 1000]
    for idx in port_list.index:
        p_point = (port_list.loc[idx].lat, port_list.loc[idx].lon)
        tpl = (idx, haversine(point, p_point))
        if tpl[1] < nearest[1]:
            nearest = tpl
    return nearest

def has_decreased_speed(args):  # sog, crit_sog
    r"""
    If the current sog is smaller than the the criterion of sog, then returns true

    Parameters
    ----------
    args : dictionary - arguments

    Returns
    -------
    boolean

    """
    if args['crit_sog'] >= args['sog']:
        return True, None
    else:
        return False, None

def get_near_port(point, semi_diameter, port_list):
    r"""
    Returns a list of ports near the point

    Parameters
    ----------
    point : list - [ latitude, longitude ]
    semi_diameter : distance criteria
    port_list : dataframe - ports information data

    Returns
    -------
    boolean

    """
    search_range = get_search_range(point, semi_diameter)

    if search_range['lon'][0] <= search_range['lon'][1]:
        near_port = port_list[
            (search_range['lat'][0] <= port_list['lat']) & (port_list['lat'] <= search_range['lat'][1]) & \
            (search_range['lon'][0] <= port_list['lon']) & (port_list['lon'] <= search_range['lon'][1])]
    else:
        near_port = port_list[(-180 <= port_list['lon']) & (port_list['lon'] <= search_range['lon'][1])]
        near_port = near_port.append(
            port_list[(search_range['lon'][0] <= port_list['lon']) & (port_list['lon'] <= 180)])
        near_port = near_port[
            (search_range['lat'][0] <= near_port['lat']) & (near_port['lat'] <= search_range['lat'][1])]

    return near_port

def has_adjacent_port(args):
    r"""
    If the current point has adjacent port, then returns true

    Parameters
    ----------
    args : dictionary - arguments

    Returns
    -------
    boolean

    """
    if not get_near_port(args['point'], args['semi_diameter'], args['port_list']).empty:
        return True, None
    else:
        return False, None

def is_changed_dest(args):  # last_dest, curr_dest
    r"""
    If current destination and last destination are defferent, then returns true

    Parameters
    ----------
    args : dictionary - arguments

    Returns
    -------
    boolean

    """
    if args['last_dest'] != args['curr_dest']:
        return True, None
    else:
        return False, None

def is_changed_stat(args):  # last_stat, curr_stat, signal
    r"""
    Returns two boolean,
    if current navigation status is not sailing and last navigation status is sailing, then returns True, False
    if current navigation status is not sailing and last navigation status is not sailing, then returns True, True
    if current navigation status is sailing and last navigation status is not sailing, then returns False, True
    if current navigation status is sailing and last navigation status is sailing, then returns False, False

    Parameters
    ----------
    args : dictionary - arguments

    Returns
    -------
    boolean, boolean - the first boolean mean

    """
    if args['signal'] == False:
        if args['last_stat'] != args['curr_stat'] and args['curr_stat'] in ['Anchoring', 'Moored'] and args[
            'last_stat'] not in ['Anchoring', 'Moored']:
            return True, False
        else:
            return False, False

    else:
        if args['last_stat'] != args['curr_stat'] and args['curr_stat'] not in ['Anchoring', 'Moored'] and args[
            'last_stat'] in ['Anchoring', 'Moored']:
            return False, True
        else:
            return True, True

def get_o_d_record(record, semi_diameter, port_list):
    r"""
    If port is identifiable at start point and end point,
    then returns a destination port, a origin port, and sailing history

    Parameters
    ----------
    record : dataframe - a single voyage record
    port_list : dataframe - ports information data

    Returns
    -------
    sailing_record : dictionary
        a dictionary about a destination port, a origin port, and sailing history data

    """
    sailing_record = {'origin': None, 'destination': None, 'nav_info': record}

    r_len = len(record)
    point = {'origin': None, 'destination': None}
    point['origin'] = [record.loc[0, 'latitude'], record.loc[0, 'longitude']]
    point['destination'] = [record.loc[r_len - 1, 'latitude'], record.loc[r_len - 1, 'longitude']]

    for i in ['origin', 'destination']:
        near_port = get_near_port(point[i], semi_diameter, port_list)

        if near_port.empty:
            return None
        else:
            idx, dist = get_nearest_port_idx(point[i], near_port)
            sailing_record[i] = port_list.loc[idx].to_dict()

    return sailing_record

def get_slicing_index(record, pre_crit, post_crit, base_score, semi_diameter, crit_sog):
    r"""
    When any of the signals are activated, it begins to search adjacent ports.

    The search continues until the navigation status changes or the sog increases more than the criterion of a sog.

    After the end of the search, if more than a certain number of signals are active,
    then the midpoint between the beginning and the end of the search is the slicing point

    Parameters
    ----------
    record : dataframe - a single voyage record
    pre_crit : list - list of signals to detect before the beginning of the search
    post_crit : list - list of signals to detect after the beginning of the search
    port_list : dataframe - ports information data
    base_score : int - minimum number of siganls that must be activated for slicing
    semi_diameter : int or float - distance criteria
    crit_sog : int or fliat - the criterion of a sog

    Returns
    -------
    slicing_index : list - list of slicing indicies

    """
    slicing_index = []
    func_dic = {'status': is_changed_stat, 'port': has_adjacent_port, \
                'destination': is_changed_dest, 'sog': has_decreased_speed}

    signal_dic = {'sog': False, 'destination': False, 'port': False, 'status': False}

    for idx in record.index[1:]:
        args_dic = {'status': {'last_stat': record.loc[idx - 1, 'nav_status'], \
                               'curr_stat': record.loc[idx, 'nav_status'], \
                               'signal': signal_dic['status']}, \
                    'port': {'point': [record.loc[idx, 'latitude'], record.loc[idx, 'longitude']], \
                             'port_list': port_list, \
                             'semi_diameter': semi_diameter}, \
                    'destination': {'last_dest': record.loc[idx - 1, 'destination'], \
                                    'curr_dest': record.loc[idx, 'destination']}, \
                    'sog': {'sog': record.loc[idx, 'sog'], \
                            'crit_sog': crit_sog}}

        if not sum(signal_dic.values()):
            for crit in pre_crit:
                if func_dic[crit](args_dic[crit])[0]:
                    signal_dic[crit] = True
            if sum(signal_dic.values()):
                start_idx = idx
            if (func_dic['status'](args_dic['status'])) == (False, True) or not func_dic['sog'](args_dic['sog'])[0]:
                if sum(signal_dic.values()) >= base_score:
                    slicing_index.append(int((idx + start_idx) / 2))
                signal_dic = {'sog': False, 'destination': False, 'port': False, 'status': False}
            else:
                for crit in post_crit:
                    if func_dic[crit](args_dic[crit])[0]:
                        signal_dic[crit] = True
        else:
            if (func_dic['status'](args_dic['status'])) == (False, True) or not func_dic['sog'](args_dic['sog'])[0]:
                if sum(signal_dic.values()) >= base_score:
                    slicing_index.append(int((idx + start_idx) / 2))
                signal_dic = {'sog': False, 'destination': False, 'port': False, 'status': False}
            else:
                for crit in post_crit:
                    if func_dic[crit](args_dic[crit])[0]:
                        signal_dic[crit] = True

    return slicing_index

def group_by_o_d(sailing_records):
    r"""
    Group sailing records by origin and destination

    Parameters
    ----------
    sailing_records : list - list of sailing record which contains o-d data

    Returns
    -------
    od_pairs : dictionary
        a dictionary about grouped sailing records by origin and destination

    """
    od_pairs = defaultdict(lambda: 0)
    for idx, record in tqdm(enumerate(sailing_records)):
        o_port = record['origin']['port_code']
        d_port = record['destination']['port_code']
        if o_port != d_port:
            if od_pairs[o_port]:
                if od_pairs[o_port][d_port]:
                    od_pairs[o_port][d_port].append(record)
                else:
                    od_pairs[o_port][d_port] = [record]
            else:
                od_pairs[o_port] = defaultdict(lambda: 0)
                od_pairs[o_port][d_port] = [record]

    return od_pairs

def _loss_method(nm, epsilon_list, min_loss_rate, min_compress_rate, min_num):
    r"""
    Data reduction based on loss rate (RDP)

    Parameters
    ----------
    nm : numpy.array - ( latitude, longitude ) of a sailing record
    epsilon_list : list - values of parameter of RDP
    min_loss_rate : float - minimum loss rate ( 0 ~ 1 )
    min_compress_rate : float - minimum data compression rate ( 0 ~ 1 )
    min_num : int - minimum number of data points that must be included in a record
        depending on the size of the starting epsilon,
        result can have a smaller number of data points than min_num.

    Returns
    -------
    mask : numpy.array - one dimentional numpy array of boolean
        i th mask data is True if i th data point should be remained
        i th mask data is False if i th data point should be removed

    """
    o_dist = get_distance_sailed(nm)
    for e in epsilon_list:
        mask = rdp(nm, epsilon=e, algo="iter", return_mask=True)
        loss_rate = 1 - np.round(get_distance_sailed(nm[mask]) / o_dist, 2)
        if loss_rate <= min_loss_rate:
            return mask
    return mask

def _comp_method(nm, epsilon_list, min_loss_rate, min_compress_rate, min_num):
    r"""
    Data reduction based on compression rate (RDP)
    _comp_method returns a mask with a minimum loss rate in the search area
    restricted by min_compress_rate

    Parameters
    ----------
    nm : numpy.array - ( latitude, longitude ) of a sailing record
    epsilon_list : list - values of parameter of RDP
    min_loss_rate : float - minimum loss rate ( 0 ~ 1 )
    min_compress_rate : float - minimum data compression rate ( 0 ~ 1 )
    min_num : int - minimum number of data points that must be included in a record
        depending on the size of the starting epsilon,
        result can have a smaller number of data points than min_num.

    Returns
    -------
    mask : numpy.array - one dimentional numpy array of boolean
        i th mask data is True if i th data point should be remained
        i th mask data is False if i th data point should be removed

    """
    o_len = len(nm)
    for e in epsilon_list:
        mask = rdp(nm, epsilon=e, algo="iter", return_mask=True)
        compress_rate = 1 - len(nm[mask]) / o_len
        if compress_rate < min_compress_rate:
            return mask
    return mask

def _num_method(nm, epsilon_list, min_loss_rate, min_compress_rate, min_num):
    r"""
    Data reduction based on a number of data points (RDP)
    _num_method returns a mask with a minimum loss rate in the search area
    restricted by min_num

    Parameters
    ----------
    nm : numpy.array - ( latitude, longitude ) of a sailing record
    epsilon_list : list - values of parameter of RDP
    min_loss_rate : float - minimum loss rate ( 0 ~ 1 )
    min_compress_rate : float - minimum data compression rate ( 0 ~ 1 )
    min_num : int - minimum number of data points that must be included in a record
        depending on the size of the starting epsilon,
        result can have a smaller number of data points than min_num.

    Returns
    -------
    mask : numpy.array - one dimentional numpy array of boolean
        i th mask data is True if i th data point should be remained
        i th mask data is False if i th data point should be removed

    """
    if len(nm) <= min_num:
        return np.array([True for i in range(len(nm))])
    for e in epsilon_list:
        mask = rdp(nm, epsilon=e, algo="iter", return_mask=True)
        if np.sum(mask) >= min_num:
            print(f'loss_rate : {1 - np.round(get_distance_sailed(nm[mask]) / get_distance_sailed(nm), 2)}')
            return mask
    return mask

def get_reduced_nav_mask(nav_info=None, epsilon_list=[2], method='loss', min_loss_rate=0.005, min_compress_rate=0.95, min_num=40):
    r"""
    Data reduction based on loss rate (RDP)

    Parameters
    ----------
    nav_info : dataframe - a sailing record
    epsilon_list : list - values of parameter of RDP
    method : str - evaluation scale to use for reduction ( 'loss', 'comp' or 'num' )
    min_loss_rate : float - minimum loss rate ( 0 ~ 1 )
    min_compress_rate : float - minimum data compression rate ( 0 ~ 1 )
    min_num : int - minimum number of data points that must be included in a record
        depending on the size of the starting epsilon,
        result can have a smaller number of data points than min_num.

    Returns
    -------
    mask : numpy.array - one dimentional numpy array
        i th mask data is True if i th data point should be remained
        i th mask data is False if i th data point should be removed

    """
    _method = {'loss': _loss_method, 'comp': _comp_method, 'num': _num_method}
    nm = nav_info[['latitude', 'longitude']].to_numpy()
    return _method[method](nm, epsilon_list, min_loss_rate, min_compress_rate, min_num)

def frechet_dist(exp_data, num_data):
    n = len(exp_data)
    m = len(num_data)
    ca = np.ones((n, m))
    ca = np.multiply(ca, -1)
    ca[0, 0] = haversine(exp_data[0], num_data[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], haversine(exp_data[i], num_data[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], haversine(exp_data[0], num_data[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                           haversine(exp_data[i], num_data[j]))
    return ca[n - 1, m - 1]

def get_distance_matrix(list_data):
    r"""
    Make frechet distance matrix to deterine similarity between paths within same o-d pair

    Parameters
    ----------
    list_data : list - a list of sailing paths

    Returns
    -------
    dist_m : numpy.array - squared matrix.
        a number of rows and columns is number opf paths grouped by a od-pair
        dist_m(i, j) = frechet distance between i th path and j th path

    """
    n_paths = len(list_data)
    n_paths
    if str(type(list_data[0])) == "<class 'numpy.ndarray'>":
        paths = list_data
    else:
        paths = []
        for i in range(n_paths):
            paths.append(list_data[i]['nav_info'][['latitude', 'longitude']].to_numpy())

    dist_m = np.zeros((n_paths, n_paths))
    for r in range(0, n_paths - 1, 1):
        for c in range(r + 1, n_paths, 1):
            dist_m[r, c] = frechet_dist(paths[r], paths[c])
            dist_m[c, r] = dist_m[r, c]

    return dist_m

def _get_not_noise_idx(result):
    r"""
    Find an index of paths that are not classified as noise

    Parameters
    ----------
    result : list - lists of labels obtained by clustering

    Returns
    -------
    not_noise_idx : list - list of index of paths that are not classified as noise

    """
    not_noise_idx = []
    for i in range(len(result)):
        if result[i] != -1:
            not_noise_idx.append(i)
    return not_noise_idx

def clustering_within_odpair(od_pairs, o_prt, d_prt, epsilon_list=[0.5], min_sample=2, silhouette_cut=0.5):
    r"""
    Clustering using DBSCAN for specific o-d pairs
    NOTICE : after noises are cleared, a silhouette score is calculated
             for ease of calculation, 2-norm was used to calculate distance

    Parameters
    ----------
    od_pairs : dictionary - a dictionary about grouped sailing records by origin and destination
    o_port : string - a port of origin
    d_port : string - a port of desitnation
    epsilon_list : list - distance criteria between points to determine whether they are neighboring points
    min_sample : int - number of minimum neighbors to be a core point
    silhouette_cut : float - minimum Silhouette score to determine cluster results are reasonable

    Returns
    -------
    dictionary - { 'first' : list, 'second' : list }
        lists in dictionary are lists of labels obtained by clustering with
        the best epsilon and the second good epsilon

    """
    scores = pd.DataFrame(None, columns=['epsilon', 'silhouette'])
    cl = DBSCAN(eps=1, min_samples=min_sample, metric='precomputed')

    dist_m = get_distance_matrix(od_pairs[o_prt][d_prt])
    for e in epsilon_list:
        cl = cl.set_params(eps=e)
        result = cl.fit_predict(dist_m)

        rd_result = result
        nn_idx = _get_not_noise_idx(result)
        rd_dist_m = dist_m[nn_idx][:, nn_idx]
        for i in range(len(result) - 1, -1, -1):
            if rd_result[i] == -1:
                rd_result = np.delete(rd_result, i)

        if len(set(rd_result)) > 1:
            scores = scores.append(
                {'epsilon': e, 'silhouette': silhouette_score(rd_dist_m, rd_result, metric='precomputed')},
                ignore_index=True)

    result = {'first': np.zeros(shape=(1, 1)), 'second': np.zeros(shape=(1, 1))}
    if scores.empty:
        print('result is None.\nPlease retry with different epsilon values or min_samples')
        return result
    else:
        scores = scores.sort_values(by='silhouette', ascending=False)
        if scores.silhouette[0] >= 0.5:
            cl = cl.set_params(eps=scores.loc[0]['epsilon'])
            result['first'] = cl.fit_predict(dist_m)
            if len(scores) > 1 and scores.silhouette[1] >= 0.5:
                cl = cl.set_params(eps=scores.loc[1]['epsilon'])
                result['second'] = cl.fit_predict(dist_m)
        return result

def remove_o_d_pairs(od_pairs, min_n_records=5):
    r"""
    Remove o-d pairs which have less than n single voyage records

    Parameters
    ----------
    od_pairs : dictionary - a dictionary about grouped sailing records by origin and destination
    min_n_records : int - minimum number of records to keep od_pair


    """
    for o_prt in tqdm(list(od_pairs.keys())):
        for d_prt in list(od_pairs[o_prt].keys()):  # *
            if len(od_pairs[o_prt][d_prt]) < min_n_records:
                del od_pairs[o_prt][d_prt]
        if not od_pairs[o_prt]:
            del od_pairs[o_prt]

def get_noiseless_label(cluster_label):
    r"""
    Sets the label of non-clustering paths to null values.

    Parameters
    ----------
    cluster_label : numpy.array - the label of the clustering of paths which have same o-d pairs

    Returns
    -------
    label : numpy.array - the label of the clustering of paths which have same o-d pairs

    """
    count = Counter(cluster_label)

    lst = []
    for k in count.keys():
        if count[k] != 1:
            lst.append(k)

    label = np.empty(len(cluster_label), dtype=np.int16)
    for i in range(len(cluster_label)):
        if cluster_label[i] in lst:
            label[i] = cluster_label[i]
        else:
            label[i] = -1
    return label

def noiseless_hierachical_clustering(dist_m, o_prt, d_prt, method=['single', 'complete', 'average', 'centroid', 'ward']):
    r"""
    Hierachical clustering for specific o-d pairs
    evaluation indicies :
        silhouette score
        silhouett score for data without noises


    Parameters
    ----------
    dist_m : numpy.array - squared matrix.
        a number of rows and columns is number opf paths grouped by a od-pair
        dist_m(i, j) = frechet distance between i th path and j th path
    o_port : string - a port of origin
    d_port : string - a port of desitnation
    method : list - methods which use for comparing distances between clusters
        available method : single, complete, average, centroid, median, ward

    Returns
    -------
    numpy.array - clustering label or zero array
    dataframe - analysis results by hyper-parameters

    """
    scores = pd.DataFrame(None, columns=['total_score'])

    uppr_dist = np.triu(dist_m)
    dists = uppr_dist[uppr_dist != 0]

    for m in method:
        link = linkage(dists, method=m)
        threshold = link[:, 2]
        for t in threshold[:-1]:
            cluster_label = fcluster(link, t=t, criterion='distance')
            noiseless_label = get_noiseless_label(cluster_label)
            mask = noiseless_label == cluster_label

            try:
                nor_sil = silhouette_score(dist_m, cluster_label, metric='precomputed')
            except:
                nor_sil = 0.1
            try:
                red_sil = silhouette_score(dist_m[mask, :][:, mask], cluster_label[mask], metric='precomputed')
            except:
                red_sil = 0.1

            scores = scores.append({'oringin': o_prt,
                                    'destination': d_prt,
                                    'method': m,
                                    'threshold': t,
                                    'normal_silhouette': nor_sil,
                                    'noiseless_silhouette': red_sil,
                                    'total_score': nor_sil + red_sil,
                                    'normal_cluster': cluster_label,
                                    'noiseless_cluster': noiseless_label}, ignore_index=True)

    s_scores = scores.sort_values('total_score', ascending=False)
    result = s_scores[s_scores['normal_silhouette'] >= 0.5]['noiseless_cluster']

    if not result.empty:
        return result.iloc[0], s_scores
    else:
        return np.zeros(shape=(1, 1)), s_scores

def is_path_cross_coastline(record, global_coastline, global_land, tolerance=2):
    r"""
    Figure out that there is an intersection between a path consisting of multiple line segments
    and the global coastlines consisting of multiple line segments or global land polygons

    global_coastlines and global_land can be downloaded from the natural earth data
    http://www.naturalearthdata.com/downloads/

    Parameters
    ----------
    record : DataFrame - a sailing record

    Returns
    -------
    bool - if a record crosses the coastlines, then return true

    """
    segments = np.concatenate(np.array([record[['longitude', 'latitude']][:-1], \
                                        record[['longitude', 'latitude']][1:]]), axis=1).reshape(-1, 2, 2)
    segments = gpd.GeoSeries(list(map(LineString, segments)))

    index_lst = list(segments.index[segments.length >= 300])  # *
    if index_lst:
        index_lst.sort()
        seg_lst = []
        if index_lst[0] == 0:
            index_lst.pop(0)
            start_idx = 1
        else:
            start_idx = 0
        for idx in index_lst:
            seg_lst.append(MultiLineString(list(segments[start_idx:idx])))
            start_idx = idx + 1
        seg_lst.append(MultiLineString(list(segments[start_idx:])))
    else:
        seg_lst = [MultiLineString(list(segments))]

    if tolerance == 0:
        for i in range(len(seg_lst)):
            if global_coastline.intersects(seg_lst[i]).any():
                return True
        return False
    else:
        for i in range(len(seg_lst)):
            for j in range(len(global_land)):
                if (MultiLineString(seg_lst[i]) & global_land[j]).length >= tolerance:
                    return True
        return False

def get_cluster_labelled_sailing_data(od_pairs, cluster_label, data_range='all'):
    # Set the label for paths in unclustered o-d pairs to -2
    for o_prt in tqdm(od_pairs.keys()):
        for d_prt in od_pairs[o_prt].keys():
            for i in [i for i in range(len(od_pairs[o_prt][d_prt]))]:
                if cluster_label[o_prt][d_prt].any():
                    od_pairs[o_prt][d_prt][i]['label'] = cluster_label[o_prt][d_prt][i]
                else:
                    od_pairs[o_prt][d_prt][i]['label'] = -2

    # Set output field
    col = ['origin', 'destination', 'sailing_path', 'sailing_time(hours)', 'clustering_label', \
           'mmsi', 'imo', 'vessel_name', 'vessel_type_code','main_vessel_type', 'middle_vessel_type',\
           'vessel_subtype','flag_code', 'length', 'width']

    dr_dic = {'all': -2, 'clustered': -1, 'non-noise': 0}

    data = pd.DataFrame(None, columns=col)

    for o_prt in tqdm(od_pairs.keys()):
        for d_prt in od_pairs[o_prt].keys():
            for i in [i for i in range(len(od_pairs[o_prt][d_prt]))]:
                od_pairs[o_prt][d_prt][i]['nav_info'] = od_pairs[o_prt][d_prt][i]['nav_info'].reset_index(drop=True)
                if od_pairs[o_prt][d_prt][i]['label'] >= dr_dic[data_range]:
                    path = np.array(od_pairs[o_prt][d_prt][i]['nav_info'][['latitude', 'longitude']])
                    time = (od_pairs[o_prt][d_prt][i]['nav_info'].dt_pos_utc[-1:].iloc[0] \
                            - od_pairs[o_prt][d_prt][i]['nav_info'].dt_pos_utc.iloc[0]).total_seconds() / 60 / 60
                    record = {'origin': o_prt, \
                              'destination': d_prt, \
                              'sailing_path': path, \
                              'sailing_time(hours)': time, \
                              'clustering_label': od_pairs[o_prt][d_prt][i]['label'], \
                              'mmsi': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'mmsi'], \
                              'imo': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'imo'], \
                              'vessel_name': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'vessel_name'], \
                              #'callsign': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'callsign'], \
                              'vessel_type_code': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'vessel_type_code'], \
                              #'vessel_class': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'vessel_class'], \
                              'main_vessel_type': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'vessel_type'], \
                              'middle_vessel_type': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'vessel_type_main'], \
                              'vessel_subtype': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'vessel_type_sub'], \
                              #'type_of_cargo': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'vessel_type_cargo'], \
                              #'flag_country': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'flag_country'], \
                              'flag_code': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'flag_code'], \
                              'length': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'length'], \
                              'width': od_pairs[o_prt][d_prt][i]['nav_info'].loc[0, 'width']}
                    data = data.append(record, ignore_index=True)

    return data

def get_representative_records(cl_sailing_data, cluster_label):
    rep_label = {}
    for o_prt in tqdm(cluster_label.keys()):
        rep_label[o_prt] = {}
        for d_prt in cluster_label[o_prt].keys():
            max_idx = -3
            if cluster_label[o_prt][d_prt].any():
                count = Counter(cluster_label[o_prt][d_prt])
                count[max_idx] = 0
                for i in count.keys():
                    if i != -1 and count[max_idx] <= count[i]:
                        max_idx = i
            rep_label[o_prt][d_prt] = max_idx

    records = cl_sailing_data
    rep_records = pd.DataFrame(None, columns=records.columns)
    for o_prt in tqdm(cluster_label.keys()):
        if o_prt in set(records['origin']):
            for d_prt in cluster_label[o_prt].keys():
                if d_prt in set(records[records['origin'] == o_prt].destination) and \
                        rep_label[o_prt][d_prt] != -3:
                    paths = records[records['origin'] == o_prt][records['destination'] == d_prt] \
                        [records['clustering_label'] == rep_label[o_prt][d_prt]].sailing_path
                    mat = get_distance_matrix(list(paths))
                    rep_records = rep_records.append(records.loc[paths.index[np.argmin(np.mean(mat, axis=0))]],
                                                     ignore_index=True)

    return rep_records

def multi_pro(sailing_records):

    #print(sailing_records[0])
    #exit()
    # 10. Group by O-D pair
    print('\n9. Group by O-D pair')
    od_pairs = group_by_o_d(sailing_records[1])



    # 11. (pre) Remove o-d pairs which have less than n single voyage records
    print('\n10. (pre) Remove o-d pairs which have less than n single voyage records')
    # proceed this function twice to speed up the analysis
    min_n_records = 5  # *

    remove_o_d_pairs(od_pairs, min_n_records)

    # 12. Load global land polygons and global coastline line segments
    print('\n11. Load global land polygons and global coastline line segments')
    land_scale = 10  # 10 or 50 or 110, 1 pixel : X meters *
    path_dir = "/Users/jeongtaegun/Desktop/workplace/현대글로벌프로젝트클래스/fileFold/map"
    worldbound_path = f'{path_dir}/global/ne_{land_scale}m_land/ne_{land_scale}m_land.shp'  # *
    worldbound = gpd.read_file(worldbound_path)
    global_land = worldbound['geometry']

    coast_scale = 110  # 10 or 50 or 110, 1 pixel : X meters *

    worldbound_path = f'{path_dir}/global/ne_{coast_scale}m_coastline/ne_{coast_scale}m_coastline.shp'  # *
    worldbound = gpd.read_file(worldbound_path)
    global_coastline = worldbound['geometry']

    # 13. Remove records across the coastline
    print('\n12. Remove records across the coastline')
    # tolerance : maximum allowable length of route which is on the land
    tolerance = 1  # unit : euclidean distance in the geographic coordinates, recommend to set within 0 to 2


    for o_prt in tqdm(list(od_pairs.keys())):  # *
        for d_prt in list(od_pairs[o_prt].keys()):  # *
            drop_lst = []
            for i in range(len(od_pairs[o_prt][d_prt])):
                if is_path_cross_coastline(od_pairs[o_prt][d_prt][i]['nav_info'], global_coastline,\
                                            global_land, tolerance=tolerance):
                    drop_lst.append(i)
            drop_lst.reverse()
            for idx in drop_lst:
                del od_pairs[o_prt][d_prt][idx]
            if not od_pairs[o_prt][d_prt]:
                del od_pairs[o_prt][d_prt]
        if not od_pairs[o_prt]:
            del od_pairs[o_prt]



    # 14. Data point reduction by RDP(Ramer Douglas Peucker) method
    print('\n13. Data point reduction by RDP(Ramer Douglas Peucker) method')
    # 14-1. Get a mask for each path created by RDP
    epsilon_list = [0.5 * np.power(1 / 2, i) for i in range(15)]  # *
    min_loss_rate = 0.05  # *
    od_masks = {}
    for o_prt in tqdm(list(od_pairs.keys())):  # *
        od_masks[o_prt] = {}
        for i, d_prt in enumerate(list(od_pairs[o_prt].keys())):  # *
            od_masks[o_prt][d_prt] = {}
            d_len = len(list(od_pairs[o_prt].keys()))
            print(f'\n ( {i} / {d_len} )')
            od_masks[o_prt][d_prt] = []
            for i in [i for i in range(len(od_pairs[o_prt][d_prt]))]:
                od_masks[o_prt][d_prt].append(
                    get_reduced_nav_mask(od_pairs[o_prt][d_prt][i]['nav_info'], epsilon_list, method='num'))

    # 14-2. Get records which are filtered by RDP
    for o_prt in tqdm(list(od_masks.keys())):  # *
        for d_prt in list(od_masks[o_prt].keys()):  # *
            for i in [i for i in range(len(od_masks[o_prt][d_prt]))]:
                od_pairs[o_prt][d_prt][i]['nav_info'] = od_pairs[o_prt][d_prt][i]['nav_info'][od_masks[o_prt][d_prt][i]]

    # 15. (pre)Remove o-d pairs which have less than n single voyage records
    print('\n14. (pre)Remove o-d pairs which have less than n single voyage records')
    # proceed this function twice to speed up the analysis
    min_n_records = 5  # *

    remove_o_d_pairs(od_pairs, min_n_records)

    # 16. Clustering
    print('\n15. Clustering')
    filtered_records = od_pairs

    '''
    # Method 1 : Clustering by DBSCAN
    min_sample = 2  # *
    silhouette_cut = 0.5
    epsilon_list = [0.1 * (i + 1) for i in range(10)]  # *
    epsilon_list.extend([1 + 0.5 * i for i in range(10)])  # *

    dbscan_label = {}
    for o_prt in tqdm(list(filtered_records.keys())):  # *
        dbscan_label[o_prt] = {}
        for d_prt in list(filtered_records[o_prt].keys()):  # *
            dbscan_label[o_prt][d_prt] = clustering_within_odpair(filtered_records, o_prt, d_prt, epsilon_list, min_sample,
                                                                        silhouette_cut)
        
    '''
    # Method 2 :  Hierachical Clustering
    hc_label = {}
    results = pd.DataFrame()
    for o_prt in tqdm(list(filtered_records.keys())):  # *
        hc_label[o_prt] = {}
        for d_prt in list(filtered_records[o_prt].keys()):  # *
            dist_m = get_distance_matrix(filtered_records[o_prt][d_prt])
            hc_label[o_prt][d_prt], result = noiseless_hierachical_clustering(dist_m, o_prt, d_prt)
            results = results.append(result, ignore_index=True)



    # 17. Get output
    print('\n16. Get output')
    # 17-1. Get cluster labelled sailing data
    cluster_label = hc_label  # DBSCAN result ( dbscan_label ) or Hierachical Clustering result ( hc_label ) *
    data_range = 'all'  # 'clustered', 'non-noise', or 'all' *

    cl_sailing_data = get_cluster_labelled_sailing_data(od_pairs, cluster_label, data_range=data_range)

    # Save cluster labelled sailing data
    path_dir = "/Users/jeongtaegun/Desktop/workplace/현대글로벌프로젝트클래스/fileFold/final_pickle"  # *
    with open(f'{path_dir}/{sailing_records[0]}_final_records.pickle', 'wb') as f: # *aqW2
        pickle.dump(cl_sailing_data, f, pickle.HIGHEST_PROTOCOL)
    # 17-2. Get representative paths
    cl_sailing_data = get_cluster_labelled_sailing_data(od_pairs, cluster_label, data_range='non-noise')
    representative_records = get_representative_records(cl_sailing_data, cluster_label)

    # Save representative paths
    path_dir = "/Users/jeongtaegun/Desktop/workplace/현대글로벌프로젝트클래스/fileFold/representative_pickle"  # *
    with open(f'{path_dir}/{sailing_records[0]}_representative_records.pickle', 'wb') as f: # *
        pickle.dump(representative_records, f, pickle.HIGHEST_PROTOCOL)
    #with open(f'{path_dir}/representative_records.pickle', 'rb') as f:
    #    representative_records = pickle.load(f)

    #with open(f'{path_dir}/final_records.pickle', 'rb') as f:
    #    final_records = pickle.load(f)

    #with open(f'{path_dir}/new_sailing_records.pickle', 'rb') as f:
    #    sailing_records = pickle.load(f)
    gc.collect()


"""
if __name__ == "__main__":

    ##############################################################################
    #                                                                            #
    #                             Excuting Section                               #
    #  When executing, the part that may need to be changed is marked with '*'.  #
    #                                                                            #
    ##############################################################################



    important_cols = ['mmsi', 'imo', 'longitude', 'latitude', 'sog', 'cog', 'destination']
    show_number_of_missing_values_in(records, important_cols)

    #path_dir = "C:/Users/SAVANNA/Dropbox/HANEUL/AIS/data2"  # *
    #with open(f'{path_dir}/records.pickle', 'wb') as f: # *
    #    pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)

    #with open(f'{path_dir}/records.pickle', 'rb') as f:
    #    records = pickle.load(f)

    # 1. Delete records which have not cetain keys that corresponds to the option
    print('\n1. Delete records which have not cetain keys that corresponds to the option')
    option = 'imo'  # 'imo', 'mmsi', or 'all' *
    delete_non_primary_key_records(records, option=option)  #

    # 2. Find and remove key values which is one-to-many relation with another key and corresponds to the option
    # option = 'all' 선택시 서로 다른 ID와 다중 연결 된 ID(mmsi, imo)를 가지는 레코드 모두 삭제
    # option = 'imo' 선택시 여러 mmsi와 다중 연결된 imo를 가지는 레코드 모두 삭제
    # option = 'mmsi' 선택시 여러 imo와 다중 연결된 mmsi를 가지는 레코드 모두 삭제
    print('\n2. Find and remove key values which is one-to-many relation with another key and corresponds to the option')
    option = 'mmsi'  # 'imo', 'mmsi', or 'all' *
    delete_multiple_linked_primary_key(records, option=option)

    # (option) Convert time series sailing records to ship-specific sailing records
    records = group_raw_data_by_ship_id(records)

    # (option) Check all navigation status
    # print(get_all_navigation_status(records))

    # 3. Sort by time series
    print('\n3. Sort by time series')
    sort_by_time_series(records)

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
        sailing_lst[i] = delete_recieving_error_data(sailing_lst[i], max_sog)

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

"""

