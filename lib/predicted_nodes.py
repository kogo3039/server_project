# -*- coding: utf-8 -*-
"""
Preprocessing and Clustering Analysis about Sailing Data

Author : HANEUL KIM
Email : rgs6827@korea.ac.kr
Python Version : 3.8.8
Package Version :
    Package Name         Version
    ------------         -------
    pandas	             1.3.1
    numpy	             1.20.3
    pyproj               3.1.0
"""
import os
import pickle
import numpy as np


def get_pickle(path_dir1, path_dir2):
    pickles1 = os.listdir(path_dir1)
    pickles2 = os.listdir(path_dir2)
    filenames1 = []
    filenames2 = []
    for pick in pickles1:
        if '.DS_Store' not in pick:
            filenames1.append(path_dir1 + '/' + pick)

    for pick in pickles2:
        if '.DS_Store' not in pick:
            filenames2.append(path_dir2 + '/' + pick)

    return filenames1, filenames2

def get_angle(vector1, vector2):
    r"""
    Returns the angle between two vectors

    Parameters
    ----------
    vector1 : list - [latitude, longitude]
    vector2 : list - [latitude, longitude]

    Returns
    -------
    angle - float - the angle between two vectors

    """
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product) * 180 / np.pi
    return angle

def get_min_dist_between_point_to_path(geodesic, point, path):
    r"""
    Returns minimum distance between the path and the point and
    the index of closest point on the path(sequential points data) to the point

    Parameters
    ----------
    point : np.array - [latitude, longitude]
    path : np.array - sequential points of sailing path

    Returns
    -------
    min_dist : float - minimum distance between the path and the point
    min_index : int - the index of closest point on the path(sequential points data) to the point

    """
    # vincenty formula ver.
    m = len(path)
    min_dist = np.inf

    for i in range(m):
        dist = geodesic.inv(point[1], point[0], path[i][1], path[i][0])[2]
        if min_dist >= dist:
            min_index = i
            min_dist = dist
    return min_dist, min_index

def get_predicted_path(geodesic, records, rep_records, O_D, point, k):
    r"""
    It is a function that receives the origin, destination, and current point and returns
    the predicted path after the current point. Find k paths closest to the current point
    among paths with same o-d pair within the clustered record.
    After that, determine the closest cluster using the distance between each path and
    the current point as a weight. If the nearest cluster is the representative path of
    the corresponding o-d pair, select that path as the predicted path. If not, select
    the closest path as the predicted path.
    After that, cut the path based on the point close to the current point and
    have a smooth connection form, and then add the current point to the path.

    Parameters
    ----------
    origin : string - origin portcode
    destination : string - destination portcode
    point : list - current point (latitude, longitude)
    k : int - Number of neighbor paths to consider

    Returns
    -------
    pre_path - np.array - predicted path

    """
    point = np.array(point)
    rel_records = records[(records['origin'] == O_D[0]) & (records['destination'] == O_D[1])]
    for index in rel_records.index:
        min_dist, min_index = get_min_dist_between_point_to_path(geodesic, point,
                                                                 rel_records.loc[index, 'sailing_path'])
        rel_records.loc[index:index + 1, 'min_dist'], rel_records.loc[index:index + 1,
                                                      'min_index'] = min_dist / 1000, min_index

    rel_records = rel_records.sort_values('min_dist').head(k)
    label = rel_records.clustering_label.unique()
    weight = dict(zip(list(label), [0 for i in range(len(label))]))
    for index in rel_records.index:
        weight[rel_records.loc[index, 'clustering_label']] = weight[rel_records.loc[index, 'clustering_label']] + 1 / \
                                                             rel_records.loc[index, 'min_dist'] ** 2

    label = max(weight, key=weight.get)

    if rep_records[(rep_records['origin'] == O_D[0]) & (rep_records['destination'] == O_D[1]) & (
            rep_records['clustering_label'] == label)].empty == False:
        # print("HERE2")
        pre_path = rep_records[(rep_records['origin'] == O_D[0]) & (rep_records['destination'] == O_D[1]) & (
                    rep_records['clustering_label'] == label)].sailing_path.iloc[0]

        # print("pre_path", pre_path)
        # print()
        dist, index = get_min_dist_between_point_to_path(geodesic, point, pre_path)
    else:

        # print("HERE")
        pre_path = rel_records[rel_records['clustering_label'] == label].head(1).sailing_path.iloc[0]
        index = int(rel_records[rel_records['clustering_label'] == label].head(1).min_index.iloc[0])
        # print("pre_path", pre_path)
        # print()

    angle = 0
    while (angle < 135 or angle > 225) and index != len(pre_path) - 1:  # * angle criteria can be changed
        vector1 = [point[0] - pre_path[index][0], point[1] - pre_path[index][1]]
        if len(pre_path) - 1 != index:
            vector2 = [pre_path[index + 1][0] - pre_path[index][0], pre_path[index + 1][1] - pre_path[index][1]]
        angle = get_angle(vector1, vector2)
        index = index + 1

    pre_path = np.append(point.reshape(-1, 2), pre_path[index - 1:], axis=0)
    print("pre_path", pre_path)
    return pre_path, rel_records.index

def load_road_optimized(geodesic, path_dir1, path_dir2, O_D, current_point, k):

    filenames1, filenames2 = get_pickle(path_dir1, path_dir2)

    for file, file2 in zip(filenames1, filenames2):
        with open(file, 'rb') as f:  # *
            records = pickle.load(f)

        with open(file2, 'rb') as f:  # *
            rep_records = pickle.load(f)

        if not records[(records['origin'] == O_D[0]) & (records['destination'] == O_D[1])].empty:
            pre_path, index = get_predicted_path(geodesic, records, rep_records, O_D, current_point, k)
        else:
            index = None
            pass


    if index is not None:
        sailing_time = records.loc[index[0], 'sailing_time(hours)']
        vessel_type = records.loc[index[0], 'main_vessel_type']
        # print(pre_path)
        return pre_path #, sailing_time
    else:
        return [], None