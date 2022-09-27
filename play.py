import argparse
import os
from numpy.lib import utils
from tqdm import tqdm
import torch
import numpy as np
from gym.gym import Gym
from model.train import Net
from collections import deque
from gym.utils import goal_enum
import torch.multiprocessing as mp
import time
from mcts.play_utils import *
seed = 5765
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)

class Play():

    def __init__(self, args):
        
        self.datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
        self.args = args
        self.net = Net(args)

        # self.gym = Gym(datapath, args)
        # self.mcts = MCTS(self.gym, self.net, args)
        self.play()


    def parallel_play(self,rank,world_size):
        iterationTrainExamples = deque([])

        # net = Net(self.args)
        # net.nnet.eval()
        gym = Gym(self.datapath, self.args)
        print("Playing with Process", rank)
        acc = 0
        matrix = {}
        stl_matrix = {}

        total = {}
        res = {}
        res_stl = {}
        for i in tqdm(range(self.args.numEps)):
            result, stl, epi = run_episode(i,gym,self.net,self.args)
            if epi in matrix:
                matrix[epi] += result
                total[epi] += 1
                stl_matrix[epi] += stl

            else:
                matrix[epi] = result
                total[epi] = 1
                stl_matrix[epi] = stl

            for key in matrix:
                res[key] = matrix[key]/total[key]
                res_stl[key] = stl_matrix[key]

            print(total,  res, res_stl)
            # if states is not None:
            #     iterationTrainExamples += states   
        save_episodes(self.args.checkpoint,iterationTrainExamples,rank) 



    def play(self):
        
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

        for ite in range(args.numIters):
            self.net.nnet.eval()

            t = time.time()  
            if self.args.parallel:          
                mp.spawn(self.parallel_play, args=(self.args.num_process,), nprocs=self.args.num_process, join=True)
            else:
                self.parallel_play(0,0)
            print(time.time() - t)

            # iterationTrainExamples += load_episodes(self.args.checkpoint) 
            # print("Number of training samples:",len(iterationTrainExamples))
            #     # print(iterationTrainExamples)
            #     # if ite==0:
            #         # self.save_episodes(iterationTrainExamples,ep)

            # print("Training....")

            # self.net.train(iterationTrainExamples)
            # print("Testing....")
            # self.net.nnet.eval()
            # self.test()


    
    def test(self):
        accuracy = 0
        for _ in tqdm(range(self.args.numEpsTest)):
            self.mcts = MCTS(self.gym, self.net, self.args)  # reset search tree
            states = run_episode(0,self.gym,self.net,self.args)
            if states is not None:
                if states[0][3]==1:
                    accuracy += 1
        print("Accuracy = ",accuracy/self.args.numEpsTest)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='111days')
    parser.add_argument('--models_folder', type=str, default='/saved_models/')
    # parser.add_argument('--model_weights', type=str, default='model_154_days_2.pt')
    parser.add_argument('--model_weights', type=str, default='model_111_days_4.pt')

    # parser.add_argument('--model_weights', type=str, default='model_goalGAILtest111_days_50.pt')

    parser.add_argument('--checkpoint', type=str, default='/episodes/')
    parser.add_argument('--load_episodes', type=bool, default=False)
    parser.add_argument('--base_path', type=str, default=os.getcwd())
    parser.add_argument('--algo', type=str, default='BC')

    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=5)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--use_trajair', type=bool, default=False)

    # parser.add_argument('--input_size', type=int, default=3)
    # parser.add_argument('--num_channels', type=int, default=1)
    # parser.add_argument('--channel_size', type=int, default=[128,256,512])
    # parser.add_argument('--kernel_size', type=int, default=3)
    # parser.add_argument('--dropout', type=float, default=0.05)
    # parser.add_argument('--balance_data', type=bool, default=True)

    parser.add_argument('--input_channels',type=int,default=3)
    parser.add_argument('--tcn_channel_size',type=int,default=512)
    parser.add_argument('--tcn_layers',type=int,default=2)
    parser.add_argument('--tcn_kernels',type=int,default=4)

    parser.add_argument('--num_context_input_c',type=int,default=2)
    parser.add_argument('--num_context_output_c',type=int,default=9)
    parser.add_argument('--cnn_kernels',type=int,default=2)

    parser.add_argument('--gat_heads',type=int, default=4)
    parser.add_argument('--graph_hidden',type=int,default=256)
    parser.add_argument('--dropout',type=float,default=0.05)
    parser.add_argument('--alpha',type=float,default=0.2)
    parser.add_argument('--cvae_hidden',type=int,default=128)
    parser.add_argument('--cvae_channel_size',type=int,default=128)
    parser.add_argument('--cvae_layers',type=int,default=2)
    parser.add_argument('--mlp_layer',type=int,default=91)


    parser.add_argument('--numMCTS', type=int, default=50)
    parser.add_argument('--cpuct', type=int, default= 1)
    parser.add_argument('--huct', type=int, default= 4000)

    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_process', type=int, default=1000)

    parser.add_argument('--numEpisodeSteps', type=int, default=30)
    parser.add_argument('--maxlenOfQueue', type=int, default=25600)
    parser.add_argument('--numEps', type=int, default=100)
    parser.add_argument('--numEpsTest', type=int, default=100)

    parser.add_argument('--numIters', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=15)

    parser.add_argument('--plot', type=bool, default=False)






    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    Play(args)
