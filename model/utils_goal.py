import math
import os
from tqdm import tqdm

from torch import nn
import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial import distance_matrix
import random



# return one-hot vector of goal position
def direction_detect(input_pos, goal,r1_flag=0):
	dir_array = torch.zeros([input_pos.shape[0],10])
	## [N, NE, E, SE, S, SW, W, NW, Takeoff, Landing]
	difference = goal-input_pos
	# print("diff",difference.shape,'goal',goal, goal.shape,"input_pos",input_pos.shape)
	for b in range(input_pos.shape[0]):
		planar_slope = np.arctan2(difference[b,1],difference[b,0])
		degrees_slope = planar_slope*180.0/np.pi

		if (goal[0]<0.9 and goal[0]> -1.00 and abs(goal[1])<0.040 and goal[2] <0.43) or r1_flag == 1: #1
			dir_array[b,8] = 1.0

		elif (goal[0]<2.200 and goal[0]> 1.000 and abs(goal[1])<0.04 and goal[2] <0.43) or r1_flag == -1:  #2,
			dir_array[b,9] = 1.0
		elif goal[2]>0.45:
			if degrees_slope <22.5 and degrees_slope >-22.5: #east
				dir_array[b,2] = 1.0
			elif degrees_slope <67.5 and degrees_slope >22.5: #NE
				dir_array[b,1] = 1.0
			elif degrees_slope <112.5 and degrees_slope >67.5: #N
				dir_array[b,0] = 1.0
			elif degrees_slope <157.5 and degrees_slope >112.5: # NW
				dir_array[b,7] = 1.0
			elif degrees_slope <-157.5 or degrees_slope >157.5: # W
				dir_array[b,6] = 1.0
			elif degrees_slope <-22.5 and degrees_slope >-67.5: #SE
				dir_array[b,3] = 1.0
			elif degrees_slope <-67.5 and degrees_slope >-112.5: #S
				dir_array[b,4] = 1.0
			elif degrees_slope <-112.5 and degrees_slope >-157.5: #SW:
				dir_array[b,5] = 1.0
		elif (goal[0]<0.9 and goal[0]> -1.00 ):
			# print("rogue r1")
			dir_array[b,8] = 1.0
		elif (goal[0]<2.200 and goal[0]> 0.900  ):
			# print("rogue r2")
			dir_array[b,9] = 1.0
	# all_zeros = not torch.any(dir_array[:])
	# if all_zeros:
		# print("caught",goal,all_zeros,dir_array[:])
	return dir_array


def get_descent_runway(goal_data,flag=0):
	## start check
	# print("hi in decent")
	# print("gola",goal_data.shape,int(goal_data[:,0].size))
	if int(goal_data[:,0].size)>10:
		# print()
		start_rate = goal_data[10,4]-goal_data[0,4]
		# print("start",start_rate)
		if start_rate > 0.0:
			# takeoff
			# print("takeoff")
			origin_goal_list=(np.where(((goal_data[:,2:4] < [0.4, 0.05]).all(axis=1))  & ((goal_data[:,2:4] > [-0.4, -0.05]).all(axis=1))))
			origin_goal_list =origin_goal_list[0]
			if origin_goal_list.size!=0 and origin_goal_list[0]<30:
				# runway origin is takeoff
				# print("2 origin_goal_list",origin_goal_list)
				end_goal_list=(np.where(((goal_data[:,2:4] < [1.7, 0.05]).all(axis=1))  & ((goal_data[:,2:4] > [1.3, -0.05]).all(axis=1))))
				end_goal_list =end_goal_list[0]
				if end_goal_list.size!=0:
					goal_loc = end_goal_list[0]
					runway_flag = 2
					# print(goal_data[:,2:5])
					# print("2 goal_loc, runway_flag",goal_loc, runway_flag)
					return goal_loc, runway_flag
			elif origin_goal_list.size!=0 and origin_goal_list[0]>30:
				# print("1 origin_goal_list",origin_goal_list)
				goal_loc = origin_goal_list[0]
				runway_flag = 1
				# print(goal_data[:,2:5])
				# print("1 goal_loc, runway_flag",goal_loc, runway_flag)
				return goal_loc, runway_flag
			# print("not origin_goal_list",origin_goal_list)
			return 0,0
		else:
			# print("not takeoff")
			return 0,0
	else:
		return 0,0

## Goal vector generation
def get_goal(goal_data):
	# goal_traj = []
	# full_traj = []
	r1_flag = 0
	origin_goal_list=(np.where(((goal_data[:,2:4] < [0.4, 0.05]).all(axis=1))  & ((goal_data[:,2:4] > [-0.4, -0.05]).all(axis=1))))
					
	origin_goal_list =origin_goal_list[0]
	flag_goal=2
	# print("goal_data",goal_data.shape)

	goal_loc, runway_flag = get_descent_runway(goal_data)
	if runway_flag == 1:

		dir_array = torch.zeros([1,10])
		dir_array[0,8] = 1.0
		full_traj =goal_data
		goal_traj = goal_data[0:goal_loc+1]
	elif runway_flag == 2:

		dir_array = torch.zeros([1,10])
		dir_array[0,9] = 1.0
		full_traj =goal_data
		goal_traj = goal_data[0:goal_loc+1]

	if  runway_flag!=1 and runway_flag!=-2:
		if int(goal_data[:,0].size) <80:
			
			goal_traj = goal_data
			full_traj =goal_data
			goal_data = goal_data[-1]
			r1_flag = 0
		else:
			full_traj =goal_data
			goal_traj = goal_data[0:-19]
			goal_data = goal_data[-20]
			r1_flag = 0
		# print("get",goal_data[2:5] )
		input_pos = np.zeros([1,3])
		dir_array = direction_detect(input_pos, goal_data[2:5],r1_flag)
	return dir_array,goal_traj, full_traj

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
		goal_data = []
		goal_data_list=[]
		for path in tqdm(all_files):
			# print(path)
			data = read_file(path, delim)
			if (len(data[:])==0):
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
				peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

				agents_in_curr_seq = np.unique(curr_seq_data[:, 1])
				self.max_agents_in_frame = max(self.max_agents_in_frame,len(agents_in_curr_seq))
				
				curr_seq_rel = np.zeros((len(agents_in_curr_seq), 3,
										 self.seq_final_len))
				curr_seq = np.zeros((len(agents_in_curr_seq), 3,self.seq_final_len ))
				goal_seq = np.zeros((len(peds_in_curr_seq), 10))
				curr_context =  np.zeros((len(agents_in_curr_seq), 2,self.seq_final_len ))
				num_agents_considered = 0
				num_peds_considered = 0
				all_data = np.concatenate( frame_data, axis=0)
#                 print(all_data.shape)
				num_peds=0
				flag =[]
				agent = -1
				for _, agent_id in enumerate(agents_in_curr_seq):
					curr_agent_seq = curr_seq_data[curr_seq_data[:, 1] ==
												 agent_id, :]


					goal_data= all_data[all_data[:,1]==
													agent_id,:]
					origin_goal_list=(np.where(((goal_data[:,2:4] < [0.1, 0.05]).all(axis=1))  & ((goal_data[:,2:4] > [-0.1, -0.05]).all(axis=1))))

					origin_goal_list =origin_goal_list[0]
#                     if origin_goal_list.size>0:
#                         print("hi",origin_goal_list[0])
#                     print(origin_goal_list.size, goal_data.shape[0])
					flag_goal=2
					if origin_goal_list.size!=0 and origin_goal_list[0]>100 :
						flag_goal = 0
						goal_data = goal_data[origin_goal_list[0]]
					if origin_goal_list.size==0:
						end_goal_list=(np.where(((goal_data[:,2:4] < [1.7, 0.05]).all(axis=1))  & ((goal_data[:,2:4] > [1.3, -0.05]).all(axis=1))))
						end_goal_list =end_goal_list[0]

						if end_goal_list.size!=0 and end_goal_list[0]>100 :
#                             print(end_goal_list[0], goal_data.shape[0])
							flag_goal = -1
							goal_data = goal_data[end_goal_list[0]]
					if  flag_goal!=0 and flag_goal!=-1:
						if int(goal_data[:,0].size) <80:
							goal_data = goal_data[-1]
						else:
							goal_data = goal_data[-40]
					goal_data= all_data[all_data[:,1]== agent_id,:]
					goal_one_hot, goal_traj, full_traj = get_goal(goal_data)

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
					if np.linalg.norm((obs[0:3,0]-obs[0:3,-1]), 2)<0.2 or np.linalg.norm((pred[0:3,0]-pred[0:3,-1]), 2)<0.2:
							flag.append(agent_id)
							num_peds-=1
							loc=int(np.where(peds_in_curr_seq == flag[0])[0])

							curr_seq_rel = np.delete(curr_seq_rel, loc, 0)
							# !!modified here to 3
							curr_seq =np.delete(curr_seq, loc, 0)
							curr_context =  np.delete(curr_context, loc, 0)
							goal_seq = np.delete(goal_seq, loc, 0)

							continue
				   
					curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq[:3,:]
					curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq[:3,:]
					curr_context[_idx,:,pad_front:pad_end] = context
					num_agents_considered += 1
					goal_seq[_idx,0:10] =goal_one_hot
					# print("goalsq",goal_seq)

				if num_agents_considered > min_agent:
					num_agents_in_seq.append(num_agents_considered)
					seq_list.append(curr_seq[:num_agents_considered])
					seq_list_rel.append(curr_seq_rel[:num_agents_considered])
					context_list.append(curr_context[:num_agents_considered])
					goal_data_list.append(goal_seq)
					# print("check",goal_data_list)
		self.num_seq = len(seq_list)
		seq_list = np.concatenate(seq_list, axis=0)
		seq_list_rel = np.concatenate(seq_list_rel, axis=0)
		context_list = np.concatenate(context_list, axis=0)
		goal_data_list = np.concatenate(goal_data_list, axis=0)
		# print("hi",goal_data_list)
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
		self.goal_position = torch.from_numpy(goal_data_list).type(torch.float)

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
			self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :], self.obs_context[start:end, :],
			self.goal_position[start:end, :]
		]
		return out

### Metrics 

def ade(y1,y2):
	"""
	y: (seq_len,2)
	"""

	loss = y1 -y2
	loss = loss**2
	loss = torch.sqrt(torch.sum(loss,1))

	return torch.mean(loss)

def fde(y1,y2):
	loss = (y1[-1,:] - y2[-1,:])**2
	return torch.sqrt(torch.sum(loss))

def rel_to_abs(obs,rel_pred):

	pred = rel_pred.copy()
	pred[0] += obs[-1]
	for i in range(1,len(pred)):
		pred[i] += pred[i-1]
	
	return pred 

def rmse(y1,y2):
	criterion = nn.MSELoss()

	# return loss
	return torch.sqrt(criterion(y1, y2))

## General utils

def read_file(_path, delim='\t'):
	data = []
	if delim == 'tab':
		delim = '\t'
	elif delim == 'space':
		delim = ' '
	with open(_path, 'r') as f:
		for line in f:
			line = line.strip().split(delim)
			line = [float(i) for i in line]
			data.append(line)
	return np.asarray(data)

def acc_to_abs(acc,obs,delta=1):
	# print("acc",acc.shape, obs.shape)
	acc = acc.permute(2,1,0)
	# print("acc2",acc.shape)
	pred = torch.empty_like(acc)
	pred[0] = 2*obs[-1] - obs[0] + acc[0]
	pred[1] = 2*pred[0] - obs[-1] + acc[1]
	
	for i in range(2,acc.shape[0]):
		pred[i] = 2*pred[i-1] - pred[i-2] + acc[i]
	return pred.permute(2,1,0)
	

def seq_collate(data):
	(obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,context_list,goal_seq_list) = zip(*data)

	_len = [len(seq) for seq in obs_seq_list]
	cum_start_idx = [0] + np.cumsum(_len).tolist()
	seq_start_end = [[start, end]
					 for start, end in zip(cum_start_idx, cum_start_idx[1:])]

	# Data format: batch, input_size, seq_len
	# LSTM input format: seq_len, batch, input_size
	obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
	pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
	obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
	pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
	context = torch.cat(context_list, dim=0 ).permute(2,0,1)
	seq_start_end = torch.LongTensor(seq_start_end)
	goal_position = torch.cat(goal_seq_list, dim=0)

	out = [
		obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end, goal_position
	]
	return tuple(out)


# def loss_func(recon_y,y,mean,log_var):
#     traj_loss = rmse(recon_y,y)
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     return traj_loss + KLD

def loss_func(recon_y,y):
	traj_loss = rmse(recon_y,y)
	return traj_loss 
