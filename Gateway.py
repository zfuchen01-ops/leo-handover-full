from Defination import EARTH_RADIUS, USER_HEIGHT
from Position import Sphere_Position


class Gateway:
    uid = 0

    def __init__(self, lon, lat, antenna_Num=1, name_str="", height=USER_HEIGHT):
        self.__class__.uid += 1
        self.id = self.uid
        self.height = height
        self.w0_pos = Sphere_Position()
        self.w0_pos.lon = lon
        self.w0_pos.lat = lat
        self.w0_pos.radius = EARTH_RADIUS + height
        self.we_pos = Sphere_Position()
        self.we_pos = self.w0_pos
        self.sat_covered = set()
        self.antenna_Num = antenna_Num
        self.name = name_str
        self.connected_sat = [None for _ in range(antenna_Num)]
