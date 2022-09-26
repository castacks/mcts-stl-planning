from gym.dataset_utils import read_file
import numpy as np
import os
import math
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
##Dataloader class

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets
    Modified from https://github.com/alexmonti19/dagnet"""
    
    def __init__(
        self, data_dir, obs_len=11, pred_len=120, skip=1,step=10,
        min_agent=0, delim=' '):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <agent_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_agent: Minimum number of agents that should be in a seqeunce
        - step: Subsampling for pred
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_agents_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.step = step
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len/self.step))
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_agents_in_seq = []
        seq_list = []
        seq_list_rel = []
        context_list = []
        for path in tqdm(all_files):
            # print(path)
            data = read_file(path, delim)
            if (len(data[:,0])==0):
                print("File is empty")
                continue
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_agents_in_frame = max(self.max_agents_in_frame,len(agents_in_curr_seq))
                
                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 3,
                                         self.seq_final_len))
                curr_seq = np.zeros((len(agents_in_curr_seq), 3,self.seq_final_len ))
                curr_context =  np.zeros((len(agents_in_curr_seq), 2,self.seq_final_len ))
                num_agents_considered = 0
                for _, agent_id in enumerate(agents_in_curr_seq):
                    curr_agent_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 agent_id, :]
                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx
                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    obs = curr_agent_seq[:,:obs_len]
                    pred = curr_agent_seq[:,obs_len+step-1::step]
                    curr_agent_seq = np.hstack((obs,pred))
                    context = curr_agent_seq[-2:,:]
                    assert(~np.isnan(context).any())
                    
                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]

                    _idx = num_agents_considered

                    if (curr_agent_seq.shape[1]!=self.seq_final_len):
                        continue

                   
                    curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq[:3,:]
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq[:3,:]
                    curr_context[_idx,:,pad_front:pad_end] = context
                    num_agents_considered += 1

                if num_agents_considered > min_agent:
                    num_agents_in_seq.append(num_agents_considered)
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    context_list.append(curr_context[:num_agents_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        context_list = np.concatenate(context_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_context = torch.from_numpy(
            context_list[:,:,:self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.max_agents = -float('Inf')
        for (start, end) in self.seq_start_end:
            n_agents = end - start
            self.max_agents = n_agents if n_agents > self.max_agents else self.max_agents

    def __len__(self):
        return self.num_seq
    
    def __max_agents__(self):
        return self.max_agents

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :], self.obs_context[start:end, :]
        ]
        return out
