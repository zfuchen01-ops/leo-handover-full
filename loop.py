import logging
import time 
import os
import argparse
from Topology import Topology
from Network import Network
from Handover import Handover
from Defination import *

from Position import Convert_Angle_to_PI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

lrout = open("./log/hyper/lr.log", 'a')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--lat_div', type=int, default=2,help='')
    args = parser.parse_args()
    start_time = time.time()
    #构建拓扑
    topo = Topology()
    orbit_num = 12
    sat_per_orbit = 12
    height = SAT_HEIGHT
    phase = 1   #轨道相位因子
    first_phi = 2.0*phase*pi/(orbit_num*sat_per_orbit)   #相邻轨道卫星相位差
    lean = 54.0/180.0*pi    #轨道倾角
    theta = 2.0*pi/orbit_num   #升交点赤经差
    topo.Add_Constellation(orbit_num,sat_per_orbit,height,first_phi,lean,theta,TYPE_2PI)

    #添加用户
    loc = []
    lat_start = -60.0
    lat_end = 60.0
    lat_div = 10
    lat_step = (lat_end-lat_start)/lat_div
    lon_start = -120
    lon_end = 120
    lon_div = 60
    lon_step = (lon_end-lon_start)/lon_div


    for j in range(lat_div):
        for i in range(lon_div):
            loc.append([Convert_Angle_to_PI(lon_start+i*lon_step),Convert_Angle_to_PI(lat_start+j*lat_step)])
    topo.Add_User_From_Input(loc)
    print("User count:%d"%len(topo.user))
    for i in range(int(len(topo.user)/2)):
        topo.user[i].User_Connect_User(topo.user[i+int(len(topo.user)/2)],'UPLOAD')

    net = Network(topo) #网络初始化
    ho = Handover(net=net)  #切换初始化

    ho_begin = 0
    ho_end = 6000
    ho_step = 30

    ho.Run_Network_Handover(ho_begin, ho_end, ho_step, 'DELAY')


    end_time = time.time()
    print(end_time-start_time)
