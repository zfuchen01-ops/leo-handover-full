from Defination import *
from Position import Sphere_Position

class User:
    uid = 0

    def __init__(self,lon, lat, height=USER_HEIGHT, elevation=USER_ELEVATION, beam=USER_BEAM):
        assert(lon>=-1*pi)
        assert(lon<=pi)
        assert(lat>=-0.5*pi)
        assert(lat<=pi/2)
        assert(height>=0)
        assert(beam>=1)
        self.__class__.uid += 1
        self.user_ID = self.uid
        self.height = height
        self.w0_pos = Sphere_Position()
        self.w0_pos.lon = lon
        self.w0_pos.lat = lat
        self.w0_pos.radius = EARTH_RADIUS+height
        self.we_pos = Sphere_Position()
        self.we_pos = self.w0_pos
        self.vel = 0.0
        self.assigned_gateway = None    # 分配的回落地面站
        self.elevation = elevation
        self.beam = beam
        self.sat_covered = set()
        self.sat_connected = None
        self.last_connected = None
        self.user_connecting_to = set()     #处于连接中的终端，该终端为源
        self.user_connecting_by = set()     #处于连接中的终端，该终端为目的
        self.user_to_connect_to = {}        #期望上行吞吐量，即星地信道容量
        self.user_to_connect_by = {}        #期望下行吞吐量，即星地信道容量
        self.allocate_band = {}     #网络中分配的带宽: {dest_user: bandwidth}
        self.allocate_dest = {}     #分配时的目的卫星ID: {dest_user: sat_id}
    
    def __hash__(self):
        return hash(self.user_ID)
    
    def __eq__(self,other):
        return self.user_ID == other.user_ID
        
    def User_Connect_User(self, dest, mode:str):
        if mode=='DOWNLOAD':
            self.user_to_connect_by[dest] = 0
            dest.user_to_connect_to[self] = 0
        elif mode=='UPLOAD':
            #期望上行吞吐量，即星地信道容量
            self.user_to_connect_to[dest] = 0
            self.allocate_band[dest] = 0
            dest.user_to_connect_by[self] = 0
        elif mode=='DUAL':
            self.user_to_connect_by[dest] = 0
            dest.user_to_connect_to[self] = 0
            self.user_to_connect_to[dest] = 0
            dest.user_to_connect_by[self] = 0
        else:
            print("Wrong mode type!!!")

    # def User_Disconnect_User(self, dest):
    #     if dest not in self.user_to_connect_to:
    #         print("Warning: Already disconnected!!!")
    #     else:
    #         self.band -= self.user_to_connect_to[dest]
    #         del self.user_to_connect_to[dest]
    #     if self not in dest.user_to_connect_by:
    #         print("Warning: Already disconnected!!!")
    #     else:
    #         dest.band -= dest.user_to_connect_by[self]
    #         del dest.user_to_connect_by[self]
