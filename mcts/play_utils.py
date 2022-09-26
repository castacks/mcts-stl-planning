
import os
from pickle import Pickler, Unpickler
from glob import glob
from collections import deque
from mcts.all_dir_monitors import goal_to_spec
from mcts.mcts import MCTS
from gym.utils import goal_enum
import numpy as np

def run_episode(rank,gym,net,args):

    curr_position , curr_start , curr_goal = gym.get_valid_start_goal()
    print("Start=", goal_enum(curr_start), " Goal=", goal_enum(curr_goal))
    STL = goal_to_spec(goal_enum(curr_goal))
    epi = goal_enum(curr_start)[0] + "-" + goal_enum(curr_goal)[0]
    epi = epi.replace('R2','R')
    epi = epi.replace('R1','R')
    old_state = []

    trainExamples = []
    episodeStep = 0

    while True:
        episodeStep += 1
        mcts = MCTS(gym, net, args)

        if args.plot: gym.plot_env(curr_position,'g',goal_position = curr_goal)

        pi = mcts.getActionProbs(curr_position, curr_goal, STL, history=old_state)
        old_state.append(curr_position)


        pi = np.squeeze(pi)
        trainExamples.append([curr_position, curr_goal, pi])

        action = np.random.choice(len(pi), p=pi)

        # action = np.argmax(pi)
        curr_position = gym.getNextState(curr_position, action)

        for value in mcts.Es.values():
            if value == 10:
                # print(mcts.temp_stack_plot)

                return 1,STL(mcts.temp_stack_plot), epi
                
        if args.plot: gym.plot_env(curr_position,'g',save=False)

        r,g = gym.getGameEnded(curr_position, curr_goal)
        if r != 0 and args.plot:
            gym.reset_plot()

        if episodeStep > args.numEpisodeSteps:
            print("Max Steps Reached")
            # if args.plot: gym.reset_plot()
            return 0,STL(mcts.temp_stack_plot), epi
            
def save_episodes(checkpoint,iterationTrainExamples,ep):
    print("Saving Episode for: ",ep)
    folder = os.getcwd() + checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, "episodes_" + str(ep) + ".examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(iterationTrainExamples)
    f.closed

def load_episodes(checkpoint):
    iterationTrainExamples = deque([])
    folder = os.getcwd() + checkpoint

    filelist = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.examples'))]
    # print(filelist,folder)
    for examplesFile in filelist:
        with open(examplesFile, "rb") as f:
            states = Unpickler(f).load()
            if states is not None:
                iterationTrainExamples += states
    print("Loaded Episodes:", len(iterationTrainExamples))
    return iterationTrainExamples