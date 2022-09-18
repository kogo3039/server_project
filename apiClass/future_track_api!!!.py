import gc
import sys
sys.path.insert(0, '../')
import pyproj
from flask_restful import Resource, reqparse
from lib.astar import Worldmaps_pacific, Worldmaps_atlantic, Astar_atlantic, Astar_pacific
from lib.predicted_nodes import load_road_optimized
from apiClass.dept_dest_code_api import CONS_FINAL_FUTURE_TRACK, CONS_REPRE_FUTURE_TRACK, \
                                        DF_FINAL_DICTIONARY, TOKEN
import warnings
warnings.filterwarnings('ignore')


class FutureTrackApi(Resource):

    def post(self):

        # data = pd.read_csv('users.csv')  # read CSV
        # data = data.to_dict()  # convert dataframe to dictionary
        parser = reqparse.RequestParser()  # initialize

        # parser.add_argument('token', required=True, help="Token cannot be blank!")  # add args
        parser.add_argument('origin', required=False, help="Origin cannot be blank!")
        parser.add_argument('destination', required=True, help="Destination cannot be blank!")
        parser.add_argument('longitude', type=float, required=True, help="Longitude cannot be blank!")
        parser.add_argument('latitude', type=float, required=True, help="Latitude cannot be blank!")
        parser.add_argument('token', required=True, help="Token cannot be blank!")
        #parser.add_argument('version')

        args = parser.parse_args()  # parse arguments to dictionary

        origin = args['origin']
        destination = args['destination']
        longitude = args['longitude']
        latitude = args['latitude']
        token = args['token']

        begin = [latitude, longitude]


        if token in TOKEN:

            dicts = {"result": "ok"}
            dicts["type"] = "FutureInfo"
            data = {}
            data['list'] = None

            if origin is not None:
                if len(origin) == 0:
                    nodes = self.astar_predict(begin, destination)
                    data['list'] = []
                    for i in range(len(nodes)):
                        if i % 4 == 0:
                            data['list'].append({"latitude": nodes[i][0], "longitude": nodes[i][1]})
                    del nodes
                    gc.collect()
                    dicts['data'] = data
                else:
                    nodes = self.odPair_predict([origin, destination], [latitude, longitude])
                    if len(nodes) == 0:
                        nodes = self.astar_predict(begin, destination)
                        data['list'] = []
                        for i in range(len(nodes)):
                            if i % 4 == 0:
                                data['list'].append({"latitude": nodes[i][0], "longitude": nodes[i][1]})
                        del nodes
                        gc.collect()
                        dicts['data'] = data
                    else:
                        data['list'] = []
                        for i in range(len(nodes)):
                            data['list'].append({"latitude": nodes[i][0], "longitude": nodes[i][1]})
                        del nodes
                        gc.collect()
                        dicts['data'] = data

            else:
                nodes = self.astar_predict(begin, destination)
                data['list'] = []
                for i in range(len(nodes)):
                    if i % 4 == 0:
                        data['list'].append({"latitude": nodes[i][0], "longitude": nodes[i][1]})
                del nodes
                gc.collect()
                dicts['data'] = data

                # create new dataframe containing new values
                # new_data = pd.DataFrame(data)
            return dicts, 200  # return data and 200 OK code
        else:
            return {"Message": "You are not authorized!!!"}, 200

    def odPair_predict(self, ods, positions):
        geodesic = pyproj.Geod(ellps='WGS84')
        pre_path = load_road_optimized(geodesic, CONS_FINAL_FUTURE_TRACK, CONS_REPRE_FUTURE_TRACK,\
                                                         ods, positions, k=3)
        del geodesic
        gc.collect()
        return pre_path

    def astar_predict(self, begin, destination):
        for i in range(DF_FINAL_DICTIONARY.shape[0]):
            if DF_FINAL_DICTIONARY['port_code'][i] == destination:
                end = [float(DF_FINAL_DICTIONARY['latitude'][i]), float(DF_FINAL_DICTIONARY['longitude'][i])]
        del DF_FINAL_DICTIONARY
        gc.collect()
        if abs(end[1] - begin[1]) < 360 - abs(end[1] - begin[1]):
            ocean_kind = "atlantic"
        else:
            ocean_kind = "pacific"

        # clr = 'red'
        # lineClr = 'orange'
        if ocean_kind == "pacific":
            print("1. start: load map")
            worldmaps = Worldmaps_pacific(begin[0], begin[1], end[0], end[1])
            print("2. start: path finding")
            astar = Astar_pacific(worldmaps)
            real_road = astar.get_route_in_pacific(ocean_kind)
            # astar.draw_map(real_road, clr, lineClr, ocean_kind)
            del worldmaps
            del astar
            gc.collect()
        elif ocean_kind == "atlantic":
            print("1. start: load map")
            worldmaps = Worldmaps_atlantic(begin[0], begin[1], end[0], end[1])
            print("2. start: path finding")
            astar = Astar_atlantic(worldmaps)
            real_road = astar.get_route_in_atlantic(ocean_kind)
            # astar.draw_map(real_road, clr, lineClr, ocean_kind)
            del worldmaps
            del astar
            gc.collect()
        return real_road