import argparse
import os
from matplotlib import pyplot as plt
from costmap import CostMap

from gym.dataset_loader import TrajectoryDataset
from gym.dataset_utils import seq_collate_old
from gym.utils import *
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy

THRESH = 8.5

class Gym():

    def __init__(self, datapath, args):

        self.datapath = datapath
        self.args = args
        self.load_action_space()
        self.costpath = args.base_path + args.dataset_folder + '111_days' + "/processed_data/train/"

        self.costmap = CostMap(self.costpath)
        if self.args.use_trajair:
            self.load_trajair()
        
        self.goal_list = goal_eucledian_list()
        self.hh = []


        if self.args.plot:
            self.fig = plt.figure()
            print("Plotting")
            self.sp = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
            self.fig_count = 0
            # plt.plot(self.traj[:,0],self.traj[:,1],'y',linewidth=10, alpha=0.2)


    def get_cost(self,curr_position,curr_goal):
        
         # input data sample
        cost = 0.0
        for i in range(3,curr_position.shape[0]):
            x = curr_position[i,0].item() #(km)
            y = curr_position[i,1].item() #(km)
            z = curr_position[i,2].item() #(km)
            yaw_diff = curr_position[i,:] - curr_position[i-3,:]
            slope = torch.atan2(yaw_diff[1],yaw_diff[0])
            if goal_enum(curr_goal) == 'R2':
                wind = 1
            else:
                wind = -1

            angle = slope*180/np.pi #degrees
            # angle = 0
            if x>-0.2 and x<1.6 and abs(y) <0.5 :
                # print("Witin")
                cost +=  -1

            # try :
            cost += self.costmap.state_value(x, y, z, angle, wind) 
            # cost += c if 

            # except:
                # cost += -1


        return cost/((curr_position.shape[0]-3))

    def load_action_space(self):

        self.traj_lib, self.index_lib = populate_traj_lib(self.args.base_path)

    def load_trajair(self):

        dataset_train = TrajectoryDataset(self.datapath + "test", obs_len=self.args.obs,
                                          pred_len=self.args.preds, step=self.args.preds_step, delim=self.args.delim)
        self.loader_train = DataLoader(
            dataset_train,  batch_size=1, num_workers=1, shuffle=True, collate_fn=seq_collate_old)

    def get_random_start_position(self,num_goals=10,p=None):

        start = torch.from_numpy(np.eye(num_goals)[np.random.choice(num_goals, 1, p = [0.16,0,0.16,0,0.16,0,0.16,0,0.18,0.18])]).float()
        # start = torch.from_numpy(np.eye(num_goals)[np.random.choice(num_goals, 1, p = [0.0,0,0.0,0,0.0,1.0,0.0,0.0,0.0,0.0])]).float()

        start_loc = goal_enum(start)[0]

        ang = np.array([90,45,0,-45,-90,-135,180,135]) + 180
        
        if start_loc !=  "R1" and start_loc != "R2":

            angle = ang[np.argmax(start)]
            angle = np.deg2rad(angle)

            z = 0.67
            x, y = -THRESH*np.cos(angle),-THRESH*np.sin(angle)


        elif start_loc == "R1":
            
            angle = 0
            z = 0.365
            angle = np.deg2rad(angle)

            x , y = 0 , 0

        elif start_loc == "R2":

            angle = 180
            z = 0.365
            angle = np.deg2rad(angle)

            x, y = 1.45, 0

        r = R.from_euler('z', angle)
        direction_matrx_rep = np.squeeze(r.as_matrix())
        trajs = (np.dot(direction_matrx_rep,self.traj_lib[2]) + (np.array([x,y,z])[:,None])).T
        return torch.from_numpy(trajs).float(), start

    def get_random_goal_location(self, start, num_goals=10,p = None):

        start_loc = goal_enum(start)[0]

        if start_loc ==  "R1" or start_loc == "R2":
            return torch.from_numpy(np.eye(num_goals)[np.random.choice(num_goals, 1, p = [0.25,0,0.25,0,0.25,0,0.25,0,0,0])]).float()
        else:
            return torch.from_numpy(np.eye(num_goals)[np.random.choice(num_goals, 1, p = [0,0,0,0,0,0,0,0,0.5,0.5])]).float()

    def get_valid_start_goal(self):
            
        while True:

            start_position, start = self.get_random_start_position()
            # self.gym.plot_env(curr_position)
            # start_position = copy.deepcopy(curr_position)
            curr_goal = self.get_random_goal_location(start)

            r,g = self.getGameEnded(start_position, curr_goal)
            if r == 0  and torch.any(start != curr_goal):
                break ##make sure start is not goal
            # else:
                # print("No viable start")
        return start_position, start, curr_goal

    def getActionSize(self):

        action_space_size = self.traj_lib.shape[0]
        return action_space_size

    def getGameEnded(self, curr_position, goal_position):
        # print(goal_position)
        for i in range(3,curr_position.shape[0]):

            current_pos = curr_position[i, :]  # check shape of traj input
            second_pos = curr_position[i-1,:] #if i>3 else current_pos ##bug at zero
            dir_array = direction_goal_detect(current_pos,second_pos)
            if (dir_array == goal_position).all(): ##wanted goal
                return 10,dir_array
            if (dir_array.any()): ##unwanted goal
                return 0, dir_array
            
        return 0 ,dir_array## no goal

    def getNextState(self, curr_position, action_choice):
        # rotate and translate action choice to end of previous executed traj
        difference = curr_position[-1, :] - curr_position[-3, :]
        difference = difference.cpu().numpy()  # verify device
        angle = np.arctan2(difference[1], difference[0])
        r = R.from_euler('z', angle)
        direction_matrx_rep = np.squeeze(r.as_matrix())
        trajs = (np.dot(direction_matrx_rep,self.traj_lib[action_choice]) + (np.array(curr_position[-1,:])[:,None])).T

    
        return torch.from_numpy(trajs).float()

    def get_hash(self, curr_position):

        # return str(curr_position[-1, 0]) + str(curr_position[-1, 1])
        return "%s-%s-%s" % (int(curr_position[-1, 0]*1000),int(curr_position[-1, 1]*1000),int(curr_position[-1, 2]*1000))

    def reset_plot(self):
        plt.close()
        self.fig = plt.figure()
        self.sp = self.fig.add_subplot(111)
        self.fig.show()
        plt.plot(self.traj[:,0],self.traj[:,1],'y',linewidth=10, alpha=0.2)


    def get_heuristic(self, curr_position, curr_goal):

        pos = self.goal_list[np.argmax(curr_goal.numpy())]
        return np.linalg.norm(curr_position[-1,:]-pos)
    
    def get_heuristic_dw(self, curr_position, curr_goal):
   
        idx_closest = np.argmin(np.linalg.norm(self.traj-np.tile(curr_position[0,:],(self.traj.shape[0],1)),axis=1))
        idx = min(idx_closest+30,self.traj.shape[0]-1)
        # print(idx_closest,idx,curr_position[-(idx-idx_closest):,:].shape)
        # print(np.mean(np.linalg.norm(curr_position[-(idx-idx_closest):,:]-self.traj[idx_closest:idx,:],axis=1)))
        # print(idx_closest,idx,self.traj.shape)
        compare_point = min(idx-idx_closest,19)
        # print(compare_point)
        return np.linalg.norm(curr_position[compare_point,:]-self.traj[idx,:])

        # return np.mean(np.linalg.norm(curr_position[-(idx-idx_closest):,:]-self.traj[idx_closest:idx,:],axis=1))
        

    def getAllNextStates(self, curr_position):
        # rotate and translate action choice to end of previous executed traj
        difference = curr_position[-1, :] - curr_position[-3, :]
        difference = difference.cpu().numpy()  # verify device
        angle = np.arctan2(difference[1], difference[0])
        r = R.from_euler('z', angle)
        direction_matrx_rep = np.squeeze(r.as_matrix())
        trajs =  np.matmul(direction_matrx_rep[None, :], self.traj_lib) + np.array(curr_position[ -1, :])[ :, None]

        return trajs

    def traj_to_action(self,pred,all_states):
        # print(pred[None,:].shape,all_states[:,:,9::10].shape)
        action_probs = np.linalg.norm(pred[None,:]-all_states[:,:,4::5],axis=1)
        # print("ha",action_probs.shape)
        action_probs = np.sum(action_probs,axis=1)
        action_probs = np.power(action_probs,-1)
        # print("action",action_probs)
        action_probs = action_probs/np.sum(action_probs)
        # print("act2",action_probs)
        # action_probs = scipy.special.softmax(np.power(action_probs,-1))
        return action_probs
    
    def plot_env(self, curr_position,color='r',save=False,goal_position=None):
        phi_1_x_r1 = [-1.2, 0.9]
        phi_1_y_r1 = [0.8, 2.5]
        phi_1_z_r1 = [0.5, 0.7]

        phi_2_x_r1 = [-3, -1.2]
        phi_2_y_r1 = [0.08, 2.5]
        phi_2_z_r1 = [0.3, 0.5]
        phi_3_x_r1 = [-3.0, 0.0]
        phi_3_y_r1 = [-0.08,0.08]
        phi_3_z_r1 = [0.3, 0.5]


        phi_1_x_r2 = [-1.5, 1.50]
        phi_1_y_r2 = [-3.0, -2.0]
        phi_1_z_r2 = [0.6, 0.8]

        phi_2_x_r2 = [4.5, 5.0]
        phi_2_y_r2 = [-3, -0.2]
        phi_2_z_r2 = [0.4, 0.6]
        phi_3_x_r2 = [1.3, 5.0]
        phi_3_y_r2 = [-0.2, 0.2]
        phi_3_z_r2 = [0.3, 0.5]
        # self.sp.grid(True)
        if color == 'r':
            self.hh.append(self.sp.plot(curr_position[:, 0], curr_position[:, 1], color=color))
        if color == 'k':
            self.hh.append(self.sp.plot(curr_position[:, 0], curr_position[:, 1], '--',color=color, linewidth=1, zorder=0))
        if color == 'c':
            self.hh.append(self.sp.plot(curr_position[:, 0], curr_position[:, 1], '--',color=color, linewidth=3, zorder=0))
        if color != 'r' and color != 'c' and color !='k':
            # self.reset_plot()
            for h in self.hh: 
                if len(h) != 0 :
                    h.pop(0).remove() 

            # for h in self.hh:
            #     
            self.sp.plot(curr_position[:, 0], curr_position[:, 1], color=color)
            if curr_position[-1,2] < 0.45:
                alt = 'r'
            elif curr_position[-1,2] > 0.7:
                alt = 'b'
            else:
                alt = 'g'    
            self.sp.scatter(curr_position[-1, 0], curr_position[-1, 1], color=alt)
        self.sp.scatter(0, 0, color='k')
        self.sp.scatter(1.45, 0, color='k')
        # if int(goal_traj[0,0,0,:,0].size) >350:
        #     self.hh.append(self.sp.plot(goal_traj[0,0,0,:350,2], goal_traj[0,0,0,:350,3], "gray", linewidth=3, alpha=0.1,zorder =1))
        # else:
        #     self.hh.append(self.sp.plot(goal_traj[0,0,0,:,2], goal_traj[0,0,0,:,3], "gray", linewidth=3, alpha=0.1,zorder = 1))



        self.hh.append(self.sp.plot([phi_1_x_r2[0],phi_1_x_r2[0],phi_1_x_r2[1],phi_1_x_r2[1],phi_1_x_r2[0]],[phi_1_y_r2[0],phi_1_y_r2[1],phi_1_y_r2[1],phi_1_y_r2[0],phi_1_y_r2[0]], c="blue", linewidth=1))
        self.hh.append(self.sp.plot([phi_2_x_r2[0],phi_2_x_r2[0],phi_2_x_r2[1],phi_2_x_r2[1],phi_2_x_r2[0]],[phi_2_y_r2[0],phi_2_y_r2[1],phi_2_y_r2[1],phi_2_y_r2[0],phi_2_y_r2[0]], c="green", linewidth=1))
        self.hh.append(self.sp.plot([phi_3_x_r2[0],phi_3_x_r2[0],phi_3_x_r2[1],phi_3_x_r2[1],phi_3_x_r2[0]],[phi_3_y_r2[0],phi_3_y_r2[1],phi_3_y_r2[1],phi_3_y_r2[0],phi_3_y_r2[0]], c="orange", linewidth=1))
        # if color == 'r':
        #     self.hh.append(self.sp.plot(curr_position[:, 0], curr_position[:, 1], color=color))
        # if color == 'g':
        #     self.reset_plot()
        #     for h in self.hh: 
        #         if len(h) != 0 :
        #             h.pop(0).remove() 

        #     # for h in self.hh:
        #     #     
        #     self.sp.plot(curr_position[:, 0], curr_position[:, 1], color=color)
        #     if curr_position[-1,2] < 0.45:
        #         alt = 'r'
        #     elif curr_position[-1,2] > 0.7:
        #         alt = 'b'
        #     else:
        #         alt = 'g'    
        #     self.sp.scatter(curr_position[-1, 0], curr_position[-1, 1], color=alt)
        # self.sp.scatter(0, 0, color='k')
        # self.sp.scatter(1.45, 0, color='k')

        plt.plot([0, 1.450], [0, 0], '--', color='k')
        plt.axis("equal")
        plt.grid(True)
        plt.xlim([-12, 12])
        plt.ylim([-12, 12])

        if save:
            plt.savefig("images/mcts_"+str(self.fig_count) + ".png")
            self.fig_count += 1
        else:
            self.fig.show()
            plt.pause(0.01)


def goal_enum(goal):
    msk = goal.squeeze().numpy().astype(bool)
    g = ["N","NE","E","SE","S","SW","W","NW","R1","R2"]
    return [g[i] for i in range(len(g)) if msk[i]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--obs', type=int, default=20)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--plot', type=bool, default=True)

    args = parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    print(goal_eucledian_list())

    gym = Gym(datapath, args)

