import cv2
from datetime import datetime
import pandas as pd
import folium
import numpy as np
from skimage import io
from haversine import haversine
from collections import defaultdict
from apiClass.dept_dest_code_api import DF_FINAL_DICTIONARY, CONST_IN_FILE_PATH, CONST_OUT_FILE_PATH


class Worldmaps_atlantic:

    def __init__(self, start_lati, start_long, end_lati, end_long):  # lat: y,  long: x
        self.origin = (start_lati, start_long)  # origin (latitude, longitude)
        self.destination = (end_lati, end_long)  # destination (latitude, longitude)
        self.equator = 234  # map 상의 적도를 지나는 y 좌표

        # 북위 84 ~ 남위 -80, 서경 -180 ~ 동경 180을 벗어나는지 체크
        if start_lati > 84 or start_lati < -80 or start_long > 180 or start_long < -180 or \
                end_lati > 84 or end_lati < -80 or end_long >= 180 or end_long < -180:
            raise Exception("Error: out of latitude/longitude ranges")

        self.atlantic_map = cv2.imread(CONST_IN_FILE_PATH + "mercato_atlantic.png", cv2.IMREAD_GRAYSCALE)
        self.route_on_atlantic_map = cv2.imread(CONST_IN_FILE_PATH + "mercato_atlantic.png")
        self.map_size = np.array(self.atlantic_map)  # map size는 1014 * 475로 동일
        self.height = self.map_size.shape[0]  # 1014
        self.width = self.map_size.shape[1]  # 475

        self.atlantic_meridian = 482  # atlantic_centered map 상에서의 본초자오선에 해당하는 x 좌표
        self.start_x_atlantic, self.start_y_atlantic = self.latilong_2_xy(self.equator, self.atlantic_meridian,\
                                                                            self.height, self.width, start_lati,\
                                                                            start_long)
        self.end_x_atlantic, self.end_y_atlantic = self.latilong_2_xy(self.equator, self.atlantic_meridian,\
                                                                        self.height, self.width, end_lati,\
                                                                        end_long)

        if self.atlantic_map[self.start_y_atlantic][self.start_x_atlantic] < 220:
            self.start_y_atlantic, self.start_x_atlantic = \
                self.coordinate_transmit(self.start_y_atlantic, self.start_x_atlantic)
            # raise Exception("Error: Non-sea coordinates (start coordinates", self.start_x_atlantic, self.start_y_atlantic, ")")
        if self.atlantic_map[self.end_y_atlantic][self.end_x_atlantic] < 220:
            self.end_y_atlantic, self.end_x_atlantic = \
                self.coordinate_transmit(self.end_y_atlantic, self.end_x_atlantic)
            # raise Exception("Error: Non-sea coordinates (end coordinates", self.end_x_atlantic, self.end_y_atlantic, ")")

        print("└. complete: load world atlantic map\n")

    def coordinate_transmit(self, cord_y, cord_x):

        print("first", cord_y, cord_x)
        minimum = 100000
        for i in range(cord_y - 5, cord_y + 6):
            for j in range(cord_x - 10, cord_x + 11):
                if i < 0 or i > 474 or j < 0 or j > 1013:
                    continue
                if 220 <= self.atlantic_map[i][j] <= 255:
                    dist = abs(i - cord_y) + abs(j - cord_x)
                    if minimum > dist:
                        minimum = dist
                        new_y, new_x = i, j

        print("second", new_y, new_x)
        return new_y, new_x

    def latilong_2_xy(self, equator, meridian, height, width, latitude, longitude):
        if latitude > 0:
            y = equator - round(latitude * (equator / 84))
        elif latitude < 0:
            y = equator + round(-latitude * ((height - equator) / 80))
        else:
            y = equator

        if longitude < 0:
            longitude = 180 + (180 + longitude)
        distance_per_long = width / 360
        x = round(meridian + distance_per_long * longitude)
        if x > width:
            x = round(meridian - (360 - longitude) * distance_per_long)

        return x, y

    # map상의 x와 y좌표를 latitude, longitude로 변환해주는 함수
    def xy_2_latilong(self, equator, meridian, height, width, x, y):
        latitude, longitude = 0, 0
        distance_per_latitude_north = 84 / equator
        distance_per_latitude_south = 80 / (height - equator)
        if y >= equator:
            latitude = (y - equator) * distance_per_latitude_south * -1
        elif y <= equator:
            latitude = (equator - y) * distance_per_latitude_north
        else:
            latitude = 0

        dateline = meridian + (width / 2)
        distance_per_longitude = 180 / (width / 2)
        if x > meridian and x <= dateline:
            longitude = (x - meridian) * distance_per_longitude
        elif x > meridian and x >= dateline:
            longitude = (180 - ((x - dateline) * distance_per_longitude)) * -1
        elif x < meridian and x <= dateline:
            longitude = (meridian - x) * distance_per_longitude * -1
        elif x < meridian and x >= dateline:
            pass  # non-case
        else:
            longitude = 0

        return latitude, longitude

    # map 상의 두 지점을 대상으로 haversine distance를 반환해주는 함수
    def xy_distance_haversine(self, start, end):
        #start_latitude, start_longitude, end_latitude, end_longitude = 0, 0, 0, 0
        start_latitude, start_longitude = self.xy_2_latilong(self.equator, self.atlantic_meridian, self.height,\
                                                            self.width,\
                                                            start[0], start[1])
        end_latitude, end_longitude = self.xy_2_latilong(self.equator, self.atlantic_meridian, self.height,\
                                                        self.width, end[0],\
                                                        end[1])
        result = haversine((start_latitude, start_longitude), (end_latitude, end_longitude), unit='km')
        return result

class Astar_atlantic:

    def __init__(self, worldmaps):

        self.worldmaps = worldmaps  # Wolrdmap 객체 (instance)
        # self.toggle_haversine = toggle_haversine # haversine을 사용할지 결정하는 toggle (Boolean)
        # self.phaselist = ["pacific", "atlantic"]
        self.step_size = 1  # 탐색 길이
        self.add = ([0, self.step_size], [0, -self.step_size], [self.step_size, 0], [-self.step_size, 0],
                    [self.step_size, self.step_size], [-self.step_size, -self.step_size],
                    [-self.step_size, self.step_size], [self.step_size, -self.step_size])


        self.atlantic_star = {'position': (self.worldmaps.start_x_atlantic, self.worldmaps.start_y_atlantic),\
                                'cost': 0, 'parent': None,\
                                'trajectory': 0}
        self.atlantic_end = {'position': (self.worldmaps.end_x_atlantic, self.worldmaps.end_y_atlantic), 'cost': 0,\
                            'parent': (self.worldmaps.end_x_atlantic, self.worldmaps.end_y_atlantic),\
                            'trajectory': 0}

        self.atlantic_opendict = defaultdict(lambda: {'cost': np.inf})
        self.atlantic_closedict = {\
            (self.worldmaps.start_x_atlantic, self.worldmaps.start_y_atlantic): {'cost': 0, 'parent': None,\
                                                                                     'trajectory': 0}}

    # A* 알고리즘을 이용하여 해역 상의 최적 경로를 탐색


    def get_route_in_atlantic(self, phase):

        start_timer = datetime.now().replace(microsecond=0)
        atlantic_road = []
        atlantic_real_road = []
        min_pos = None

        while min_pos != self.atlantic_end['position']:

            #print("log: {} {} {}".format(self.atlantic_star['position'], min_pos, self.atlantic_end['position']))
            s_point = list(self.atlantic_closedict)[-1]
            for i in range(len(self.add)):
                x = s_point[0] + self.add[i][0]
                if x < 0 or x >= self.worldmaps.width:
                    continue
                y = s_point[1] + self.add[i][1]
                if y < 0 or y >= self.worldmaps.height:
                    continue
                if (x, y) in self.atlantic_closedict.keys():
                    continue
                if self.worldmaps.atlantic_map[y, x] < 220:  # 220 지도상 이동 가능한 바다인지 판별
                    continue

                G = self.atlantic_closedict[s_point]["trajectory"] + self.worldmaps.xy_distance_haversine((x, y),s_point)  # trajectory-based + haversine distacne G
                H = self.worldmaps.xy_distance_haversine(self.atlantic_end['position'], (x, y))  # haversine distance H
                F = G + H

                if self.atlantic_opendict[(x, y)]['cost'] > F:
                    self.atlantic_opendict[(x, y)] = {'cost': F, 'parent': s_point, 'trajectory': G}

            # 비교 탐색 향후 개선 필
            min_val = {'cost': np.inf}
            for pos, val in self.atlantic_opendict.items():
                if val['cost'] < min_val['cost']:  # Search minimal cost in openlist
                    min_val = val
                    min_pos = pos
            self.atlantic_closedict[min_pos] = min_val
            self.atlantic_opendict.pop(min_pos)

        pos = min_pos
        atlantic_road.append(pos)
        atlantic_real_road.append(self.worldmaps.xy_2_latilong(self.worldmaps.equator,
                                                               self.worldmaps.atlantic_meridian,
                                                               self.worldmaps.height,
                                                               self.worldmaps.width,
                                                               pos[0], pos[1]))

        while self.atlantic_closedict[pos]['parent'] != None:
            pos = self.atlantic_closedict[pos]['parent']
            # self.atlantic_route_length += self.atlantic_closedict[pos]['trajectory']
            atlantic_road.append(pos)
            atlantic_real_road.append(self.worldmaps.xy_2_latilong(self.worldmaps.equator,
                                                                   self.worldmaps.atlantic_meridian,
                                                                   self.worldmaps.height,
                                                                   self.worldmaps.width,
                                                                   pos[0], pos[1]))

        end_timer = datetime.now()
        cal_time = end_timer - start_timer
        print("└. complete: select route on", phase, "centered map (calculation time =", cal_time, ")\n")

        return atlantic_real_road

    def draw_map(self, real_road, clr, lineClr, phase):

        seaMap = folium.Map(location=[35, 123], zoom_start=5)
        if phase:
            pre_path = [
                (real_road[i][0], real_road[i][1]) if 0 <= real_road[i][1] <= 180 \
                    else (real_road[i][0], real_road[i][1]) for i in range(len(real_road))
            ]

        for i in range(len(pre_path)):
            path = list(pre_path[i])
            if i == 0:
                folium.Marker(location=path,
                              zoom_start=17,
                              icon=folium.Icon(color=clr, icon='star'),
                              popup="lat:{0:0.2f} || lon:{1:0.2f}".format(path[0], path[1])
                              ).add_to(seaMap)
            else:
                folium.CircleMarker(location=path,
                                    zoom_start=17,
                                    color=clr,
                                    fill_color=clr,
                                    popup="lat:{0:0.2f} || lon:{1:0.2f}".format(path[0], path[1]),
                                    radius=5,
                                    ).add_to(seaMap)

        folium.PolyLine(locations=pre_path, color=lineClr, tooltip='PolyLine').add_to(seaMap)
        seaMap.save(CONST_OUT_FILE_PATH + 'Astar*loadMap.html')
        print("LoadMap Completion!")

class Worldmaps_pacific:

    def __init__(self, start_lati, start_long, end_lati, end_long): # lat: y,  long: x
        self.origin = (start_lati, start_long) # origin (latitude, longitude)
        self.destination = (end_lati, end_long) # destination (latitude, longitude)
        self.equator = 234  # map 상의 적도를 지나는 y 좌표

        # 북위 84 ~ 남위 -80, 서경 -180 ~ 동경 180을 벗어나는지 체크
        if start_lati > 84 or start_lati < -80 or start_long > 180 or start_long < -180 or \
                end_lati > 84 or end_lati < -80 or end_long >= 180 or end_long < -180:
            raise Exception("Error: out of latitude/longitude ranges")

        self.pacific_map = cv2.imread(CONST_IN_FILE_PATH + "mercato_pacific.png", cv2.IMREAD_GRAYSCALE)
        self.route_on_pacific_map = cv2.imread(CONST_IN_FILE_PATH + "mercato_pacific.png")
        self.map_size = np.array(self.pacific_map)  # map size는 1014 * 475로 동일
        self.height = self.map_size.shape[0]  # 1014
        self.width = self.map_size.shape[1]  # 475

        self.pacific_meridian = 74  # pacific_centered map 상에서의 본초자오선에 해당하는 x 좌표
        self.start_x_pacific, self.start_y_pacific = self.latilong_2_xy(self.equator, self.pacific_meridian,\
                                                                        self.height, self.width, start_lati,\
                                                                        start_long)
        self.end_x_pacific, self.end_y_pacific = self.latilong_2_xy(self.equator, self.pacific_meridian,\
                                                                    self.height, self.width, end_lati, end_long)

        if self.pacific_map[self.start_y_pacific][self.start_x_pacific] < 220:
            self.start_y_pacific, self.start_x_pacific = \
                self.coordinate_transmit(self.start_y_pacific, self.start_x_pacific)
            # raise Exception("Error: Non-sea coordinates (start coordinates", self.start_x_pacific, self.start_y_pacific, ")")
        if self.pacific_map[self.end_y_pacific][self.end_x_pacific] < 220:
            self.end_y_pacific, self.end_x_pacific = \
                self.coordinate_transmit(self.end_y_pacific, self.end_x_pacific)
            # raise Exception("Error: Non-sea coordinates (end coordinates", self.end_x_pacific, self.end_y_pacific, ")")

        print("└. complete: load world map\n")

    def coordinate_transmit(self, cord_y, cord_x):
        print("first", cord_y, cord_x)
        minimum = 100000
        for i in range(cord_y - 5, cord_y + 6):
            for j in range(cord_x - 10, cord_x + 11):
                if i < 0 or i > 474 or j < 0 or j > 1013:
                    continue
                if 220 <= self.pacific_map[i][j] <= 255:
                    dist = abs(i-cord_y) + abs(j-cord_x)
                    if minimum > dist:
                        minimum = dist
                        new_y, new_x = i, j

        print("second", new_y, new_x)
        return new_y, new_x


    def latilong_2_xy(self, equator, meridian, height, width, latitude, longitude):
        if latitude > 0:
            y = equator - round(latitude * (equator / 84))
        elif latitude < 0:
            y = equator + round(-latitude * ((height - equator) / 80))
        else:
            y = equator

        if longitude < 0:
            longitude = 180 + (180 + longitude)
        distance_per_long = width / 360
        x = round(meridian + distance_per_long * longitude)
        if x > width:
            x = round(meridian - (360 - longitude) * distance_per_long)

        return x, y

    # map상의 x와 y좌표를 latitude, longitude로 변환해주는 함수
    def xy_2_latilong(self, equator, meridian, height, width, x, y):
        latitude, longitude = 0, 0
        distance_per_latitude_north = 84 / equator
        distance_per_latitude_south = 80 / (height - equator)
        if y >= equator:
            latitude = (y - equator) * distance_per_latitude_south * -1
        elif y <= equator:
            latitude = (equator - y) * distance_per_latitude_north
        else:
            latitude = 0

        dateline = meridian + (width / 2)
        distance_per_longitude = 180 / (width / 2)
        if x > meridian and x <= dateline:
            longitude = (x - meridian) * distance_per_longitude
        elif x > meridian and x >= dateline:
            longitude = (180 - ((x - dateline) * distance_per_longitude)) * -1
        elif x < meridian and x <= dateline:
            longitude = (meridian - x) * distance_per_longitude * -1
        elif x < meridian and x >= dateline:
            pass  # non-case
        else:
            longitude = 0

        return latitude, longitude

    # map 상의 두 지점을 대상으로 haversine distance를 반환해주는 함수
    def xy_distance_haversine(self, start, end):
        #start_latitude, start_longitude, end_latitude, end_longitude = 0, 0, 0, 0

        start_latitude, start_longitude = self.xy_2_latilong(self.equator, self.pacific_meridian, self.height, self.width,\
                                                                 start[0], start[1])
        end_latitude, end_longitude = self.xy_2_latilong(self.equator, self.pacific_meridian, self.height, self.width, end[0],\
                                                             end[1])
        result = haversine((start_latitude, start_longitude), (end_latitude, end_longitude), unit='km')
        return result

class Astar_pacific:

    def __init__(self, worldmaps):

        self.worldmaps = worldmaps # Wolrdmap 객체 (instance)
        #self.toggle_haversine = toggle_haversine # haversine을 사용할지 결정하는 toggle (Boolean)
        #self.phaselist = ["pacific", "atlantic"]
        self.step_size = 1  # 탐색 길이
        self.add = ([0, self.step_size], [0, -self.step_size], [self.step_size, 0], [-self.step_size, 0],
                    [self.step_size, self.step_size], [-self.step_size, -self.step_size],
                    [-self.step_size, self.step_size], [self.step_size, -self.step_size])

        self.pacific_star = {'position': (self.worldmaps.start_x_pacific, self.worldmaps.start_y_pacific), 'cost': 0,\
                             'parent': None, 'trajectory': 0}
        self.pacific_end = {'position': (self.worldmaps.end_x_pacific, self.worldmaps.end_y_pacific), 'cost': 0,\
                            'parent': (self.worldmaps.end_x_pacific, self.worldmaps.end_y_pacific), 'trajectory': 0}

        self.pacific_opendict = defaultdict(lambda: {'cost': np.inf})  # A* 탐색을 위한 open set (list)
        self.pacific_closedict = {(self.worldmaps.start_x_pacific, self.worldmaps.start_y_pacific): {'cost': 0, \
                                 'parent': None,'trajectory': 0}}  # A* 탐색을 위한 close set (list)

    # A* 알고리즘을 이용하여 해역 상의 최적 경로를 탐색
    def get_route_in_pacific(self, phase):

        start_timer = datetime.now().replace(microsecond=0)
        pacific_road = []
        pacific_real_road = []
        min_pos = None

        while min_pos != self.pacific_end['position']:
           # print("log: {} {} {}".format(self.pacific_star['position'], min_pos, self.pacific_end['position']))
            s_point = list(self.pacific_closedict)[-1]
            for i in range(len(self.add)):
                x = s_point[0] + self.add[i][0]
                if x < 0 or x >= self.worldmaps.width:
                    continue
                y = s_point[1] + self.add[i][1]
                if y < 0 or y >= self.worldmaps.height:
                    continue
                if (x, y) in self.pacific_closedict.keys():
                    continue
                if self.worldmaps.pacific_map[y, x] < 220:  # 220 지도상 이동 가능한 바다인지 판별
                    continue

                G = self.pacific_closedict[s_point]["trajectory"] + self.worldmaps.xy_distance_haversine((x, y), s_point)  # trajectory-based + haversine distacne G
                H = self.worldmaps.xy_distance_haversine(self.pacific_end['position'], (x, y))  # haversine distance H
                F = G + H

                if self.pacific_opendict[(x, y)]['cost'] > F:
                    self.pacific_opendict[(x, y)] = {'cost': F, 'parent': s_point, 'trajectory': G}

            # 비교 탐색 향후 개선 필
            min_val = {'cost': np.inf}
            for pos, val in self.pacific_opendict.items():
                if val['cost'] < min_val['cost']:  # Search minimal cost in openlist
                    min_val = val
                    min_pos = pos
            self.pacific_closedict[min_pos] = min_val
            self.pacific_opendict.pop(min_pos)

        pos = min_pos
        pacific_road.append(pos)
        pacific_real_road.append(self.worldmaps.xy_2_latilong(self.worldmaps.equator,
                                                              self.worldmaps.pacific_meridian,
                                                              self.worldmaps.height,
                                                              self.worldmaps.width,
                                                              pos[0], pos[1]))

        while self.pacific_closedict[pos]['parent'] != None:
            pos = self.pacific_closedict[pos]['parent']
            pacific_road.append(pos)
            pacific_real_road.append(self.worldmaps.xy_2_latilong(self.worldmaps.equator,
                                                                  self.worldmaps.pacific_meridian,
                                                                  self.worldmaps.height,
                                                                  self.worldmaps.width,
                                                                  pos[0], pos[1]))

        end_timer = datetime.now()
        cal_time = end_timer - start_timer
        print("└. complete: select route on", phase, "centered map(calculation time =", cal_time, ")\n")

        return pacific_real_road

    def draw_map(self, real_road, clr, lineClr, phase):

        seaMap = folium.Map(location=[35, 123], zoom_start=5)
        if phase:
            pre_path = [
                (real_road[i][0], real_road[i][1]) if 0 <= real_road[i][1] <= 180 \
                    else (real_road[i][0], real_road[i][1]+360) for i in range(len(real_road))
                ]

        for i in range(len(pre_path)):
            path = list(pre_path[i])
            if i == 0:
                folium.Marker(location=path,
                                zoom_start=17,
                                icon=folium.Icon(color=clr, icon='star'),
                                popup="lat:{0:0.2f} || lon:{1:0.2f}".format(path[0], path[1])
                                 ).add_to(seaMap)
            else:
                folium.CircleMarker(location=path,
                                    zoom_start=17,
                                    color=clr,
                                    fill_color=clr,
                                    popup="lat:{0:0.2f} || lon:{1:0.2f}".format(path[0], path[1]),
                                    radius=5,
                                    ).add_to(seaMap)

        folium.PolyLine(locations=pre_path, color=lineClr, tooltip='PolyLine').add_to(seaMap)
        seaMap.save(CONST_OUT_FILE_PATH+'Astar*loadMap.html')
        print("LoadMap Completion!")

if __name__ == "__main__":

    start = [36.97253667, 126.83001333]
    destination = "PATBG"
    for i in range(DF_FINAL_DICTIONARY.shape[0]):
        if DF_FINAL_DICTIONARY['port_code'][i] == destination:
            end = [float(DF_FINAL_DICTIONARY['latitude'][i]), float(DF_FINAL_DICTIONARY['longitude'][i])]


    if abs(end[1]-start[1]) < 360 - abs(end[1]-start[1]):
        ocean_kind = "atlantic"
    else:
        ocean_kind = "pacific"

    clr = 'red'
    lineClr = 'orange'
    if ocean_kind == "pacific":
        print("1. start: load map")
        worldmaps = Worldmaps_pacific(start[0], start[1], end[0], end[1])
        print("2. start: path finding")
        astar = Astar_pacific(worldmaps)
        real_road = astar.get_route_in_pacific(ocean_kind)
        astar.draw_map(real_road, clr, lineClr, ocean_kind)
    elif ocean_kind == "atlantic":
        print("1. start: load map")
        worldmaps = Worldmaps_atlantic(start[0], start[1], end[0], end[1])
        print("2. start: path finding")
        astar = Astar_atlantic(worldmaps)
        real_road = astar.get_route_in_atlantic(ocean_kind)
        astar.draw_map(real_road, clr, lineClr, ocean_kind)

