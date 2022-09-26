import numpy as np
import scipy
import math
import torch
from tqdm import tqdm
from line_profiler import LineProfiler
import time
from matplotlib import pyplot as plt
from mcts.stl_specs import monitor_R2
from collections import deque
EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, gym, nnet, args):

        self.gym = gym
        self.nnet = nnet
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(self.device)
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.Hs = {}
        self.Hsa = {}

        self.stack = deque()

    def getActionProbs(self, curr_position, goal_postion, STL, temp=1, max_time = None, history=None):
        self.history = history

        # return np.eye(self.gym.getActionSize())[np.random.choice(self.gym.getActionSize(), 1)]

        # for i in tqdm(range(self.args.numMCTS), desc="MCTS Trees"):
        start_time = time.time()
        if max_time is None:
            for i in (range(self.args.numMCTS)):
                # print("MCTS Tree #" + str(i))
                self.search(curr_position, goal_postion,0, STL)
        else:
            while (time.time()-start_time) < max_time:
                self.search(curr_position, goal_postion,0, STL)


        s = self.gym.get_hash(curr_position)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.gym.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if int(counts_sum) != 0:
            probs = [x / counts_sum for x in counts]
        else:
            # print(self.Nsa,self.Qsa)
            print("All counts zero for ",s)
            probs = np.ones_like(counts)/self.gym.getActionSize()
            # probs = None
        return probs

    # @profile
    def search(self, curr_position, goal_position,h, STL):
        s = self.gym.get_hash(curr_position)
        if s not in self.Es:
            self.Es[s],_ = self.gym.getGameEnded(curr_position, goal_position)

        if self.Es[s] != 0:
            # terminal node
            # print("Terminal Node")
            return self.Es[s],h

        if s not in self.Ps:
            # leaf node
            v =  self.gym.get_cost(curr_position,goal_position) 
            # v = -1
            if self.stack:
                self.temp_stack_plot = np.concatenate(self.stack, axis=0)
            if self.args.plot: self.gym.plot_env(curr_position, 'r')

            curr_position = curr_position.to(self.device) ##km to m
            goal_position = goal_position.to(self.device)

            pred = self.nnet.predict(curr_position, goal_position)
            if self.args.plot: self.gym.plot_env(np.transpose(pred),'k')
            all_next_states = (self.gym.getAllNextStates(curr_position.cpu())) # added a copy to cpu since Allnextstates also performs numpy operations
            self.Ps[s] = torch.clamp(torch.from_numpy(self.gym.traj_to_action(pred[:,:4],all_next_states)), max=0.5)

            self.Ns[s] = 0
            # print("Leaf")
            for _ in range(len(self.stack)):
                # print("poppin")
                self.stack.pop()
            return v,h

        cur_best = -float('inf')
        best_act = -1
        heu = np.zeros(self.gym.getActionSize())
        for a in range(self.gym.getActionSize()):

            next_state = self.gym.getNextState(curr_position,a)

            if self.history:
                temp_old_state = np.concatenate(self.history, axis=0)
                if self.stack:
                    temp_stack_state = np.concatenate(self.stack, axis=0)
                    stack_added_state = np.concatenate((temp_old_state,temp_stack_state))

                    temp_state = np.concatenate((stack_added_state,next_state))
                else:
                    temp_state = np.concatenate((temp_old_state,next_state))
                temp_state = temp_state[::5,:]
                heu[a] =  STL(temp_state)

            else:

                if self.stack:
                    temp_stack_state = np.concatenate(self.stack, axis=0)

                    temp_state = np.concatenate((temp_stack_state,next_state))

                    temp_state = temp_state[::5,:]

                    heu[a] =  STL(temp_state)

                else:
                    heu[a] =  STL(next_state)



                


        heu = heu / np.linalg.norm(heu)
        heu = scipy.special.softmax(heu)
        # pick the action with the highest upper confidence bound
        for a in range(self.gym.getActionSize()):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)]  + self.args.cpuct * self.Ps[s][a] * (math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])) + self.args.huct*self.Hsa[(s, a)] 
            else:
                u =  self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  + self.args.huct*heu[a] 

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act

        next_position = self.gym.getNextState(curr_position, a)

        self.stack.append(next_position)

        v , h = self.search(next_position, goal_position,heu[a], STL)
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
            self.Hsa[(s, a)] = (self.Nsa[(s, a)] * self.Hsa[(s, a)] + h) / (self.Nsa[(s, a)] + 1)


        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            self.Hsa[(s, a)] = h

        self.Ns[s] += 1
        return v,h
