import math
import os
from tqdm import tqdm

from torch import nn
import torch
from torch.utils.data import Dataset
import numpy as np
import random
THRESH = 9 #KM


def populate_traj_lib(path):
	
    # note position of motion prim library text files
    lib_path = path + '/gym/traj_lib_0SI_new.txt'
    index_path = path + '/gym/traj_index_0SI_new.txt'
    print("Loading traj lib from", lib_path)
    ## obtain a 3d matrix of each trajectory's (x, y, z) positions into an array
    file1 = open(lib_path, 'r',newline='\r\n')
    traj_no = 30 # note the number
    count, j = 0,0
    traj_lib = np.zeros([traj_no,3,20])

    for line in open(lib_path, 'r',newline='\n'):
        if count%4 == 1:
            traj_lib[j,0] = np.fromstring( line.strip(), dtype=float, sep=' ' )/1000
        if count%4 == 2:
            traj_lib[j,1] = np.fromstring( line.strip(), dtype=float, sep=' ' )/1000
        if count%4 == 3:
            traj_lib[j,2] = np.fromstring( line.strip(), dtype=float, sep=' ' )/1000
            j+=1
        count += 1

    ## obtain the details of each trajectory from the index
    file1 = open(index_path, 'r',newline='\r\n')

    j = 0
    index_lib = np.zeros([traj_no,6])

    for line in open(index_path, 'r',newline='\n'):
        index_lib[j,:] = np.fromstring( line.strip(), dtype=float, sep=' ' )
        j+=1
    
    return traj_lib,index_lib


    # return one-hot vector of goal position
def direction_goal_detect(pos,second_pos):
    
    dir_array = torch.zeros([10]) ## [N, NE, E, SE, S, SW, W, NW, R1, R2]
    yaw_diff = pos-second_pos

    if np.linalg.norm(pos) > THRESH  :
            planar_slope = torch.atan2(pos[1],pos[0])
            degrees_slope = planar_slope*180.0/np.pi

            if degrees_slope <22.5 and degrees_slope >-22.5: #east
                dir_array[2] = 1.0
            elif degrees_slope <67.5 and degrees_slope >22.5: #NE
                dir_array[1] = 1.0
            elif degrees_slope <112.5 and degrees_slope >67.5: #N
                dir_array[0] = 1.0
            elif degrees_slope <157.5 and degrees_slope >112.5: # NW
                dir_array[7] = 1.0
            elif degrees_slope <-157.5 or degrees_slope >157.5: # W
                dir_array[6] = 1.0
            elif degrees_slope <-22.5 and degrees_slope >-67.5: #SE
                dir_array[3] = 1.0
            elif degrees_slope <-67.5 and degrees_slope >-112.5: #S
                dir_array[4] = 1.0
            elif degrees_slope <-112.5 and degrees_slope >-157.5: #SW:
                dir_array[5] = 1.0
        # print("Outer pos reached",goal_enum(dir_array))
    else:
        
            yaw_diff_slope = torch.atan2(yaw_diff[1],yaw_diff[0])
            yaw_diff_slope_degrees = yaw_diff_slope*180.0/np.pi
            # print(yaw_diff_slope_degrees)
            if pos[0]<0.2 and pos[0]> -0.2 and abs(pos[1])<0.20 and pos[2] <0.7: #1
                if abs(yaw_diff_slope_degrees) <20.0:
                    dir_array[8] = 1.0
                    return dir_array
                    # print("Runway reached",goal_enum(dir_array))


            elif pos[0]<1.7 and pos[0]> 1.5 and abs(pos[1])<0.2 and pos[2] <0.7:  #2,
                # print("bad head",abs(yaw_diff_slope_degrees))
                if 180-abs(yaw_diff_slope_degrees) <20.0:
                    dir_array[9] = 1.0
                    # print("good head")

                    return dir_array
                    # print("Runway reached",goal_enum(dir_array))

    return dir_array

def goal_enum(goal):
    msk = goal.squeeze().numpy().astype(bool)
    g = ["N","NE","E","SE","S","SW","W","NW","R1","R2"]
    return [g[i] for i in range(len(g)) if msk[i]]

def goal_eucledian_list(num_goals = 10):

    pos = []
    for goal_idx in range(num_goals): 
        ang = np.array([90,45,0,-45,-90,-135,180,135])
        
        if goal_idx < 8:
            pos.append(np.array([(THRESH+1)*np.cos(np.deg2rad(ang[goal_idx])),(THRESH+1)*np.sin(np.deg2rad(ang[goal_idx])), 1.0 ]))
        elif goal_idx == 8:
            pos.append(np.array([0.0,0.0,0.2]))
        elif goal_idx == 9:
           pos.append(np.array([-3.0,-3.5,0.8]))

    return pos

def get_ref_hand_traj():

    x = np.arange(-3.0,3.0,36.321/1000)
    y = np.repeat(-2,len(x))
    z = np.repeat(0.6,len(x))

    traj = np.vstack((x,y,z))

    y = np.arange(-2,0.0,36.321/1000)
    x = np.repeat(3.0,len(y))
    z = np.repeat(0.5,len(x))
    # print(traj.shape,np.vstack((x,y,z)).transpose().shape)
    traj = np.hstack((traj,np.vstack((x,y,z))))
    # print(traj.shape,np.vstack((x,y,z)).transpose().shape)

    x = np.flip(np.arange(0.0,3.0,30.321/1000))
    y = np.repeat(0.0,len(x))
    z = np.repeat(0.35,len(x))

    traj = np.hstack((traj,np.vstack((x,y,z))))

    return traj.transpose()

def get_ref_exp_traj():

    traj = np.genfromtxt('/home/jay/AITF/aitf_ws/src/aitf_mcts/gym/2799.csv',delimiter=",")

    return traj
    