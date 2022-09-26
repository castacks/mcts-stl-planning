import torch
from torch import nn

from model.tcn_model import TemporalConvNet
from model.gat_model import GAT
from model.cvae_base import CVAE
from model.utils_goal import acc_to_abs




class TrajAirNetGoalGAIL(nn.Module):
	def __init__(self, args):
		super(TrajAirNetGoalGAIL, self).__init__()

		input_size = args.input_channels
		n_classes = int(args.preds/args.preds_step)
		num_channels= [args.tcn_channel_size]*args.tcn_layers
		num_channels.append(n_classes)
		tcn_kernel_size = args.tcn_kernels
		dropout = args.dropout
		
		graph_hidden = args.graph_hidden
		gat_in = n_classes*args.obs+args.num_context_output_c
		gat_out = n_classes*args.obs+args.num_context_output_c
		n_heads = args.gat_heads 
		alpha = args.alpha
		
		cvae_encoder = [n_classes*n_classes]
		for layer in range(args.cvae_layers):
			cvae_encoder.append(args.cvae_channel_size)
		cvae_decoder = [args.cvae_channel_size]*args.cvae_layers
		cvae_decoder.append(input_size*args.mlp_layer)

	 

		self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
		self.tcn_encoder_y = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
		self.cvae = CVAE(encoder_layer_sizes = cvae_encoder,latent_size = args.cvae_hidden, decoder_layer_sizes =cvae_decoder,conditional=True, num_labels= gat_out+gat_in)
		self.gat = GAT( nin=gat_in, nhid = graph_hidden, nout = gat_out, alpha = alpha, nheads = n_heads)
		self.linear_decoder = nn.Linear(args.mlp_layer,n_classes)
		self.context_conv = nn.Conv1d(in_channels=args.num_context_input_c, out_channels=1, kernel_size=args.cnn_kernels)
		self.context_linear = nn.Linear(args.obs-1,args.num_context_output_c)
		self.relu = nn.ReLU()
		self.goal_expand = nn.Linear(10,128)

		self.init_weights()

	def init_weights(self):
		self.linear_decoder.weight.data.normal_(0, 0.05)
		self.context_linear.weight.data.normal_(0, 0.05)
		self.context_conv.weight.data.normal_(0, 0.1)
		
	def forward(self, x, y, adj,context,goal_vector,sort=False):        
		
		encoded_trajectories_x = []
		encoded_appended_trajectories_x = []
		encoded_trajectories_y = []
		latent_space_appended = []
		# pass all agents through encoder

		# for agent in range(x.shape[2]):
		# print("x,", x.shape)
			
		x1 = torch.transpose(x, 0, 2)
		# print("x1,", x1.shape)
		encoded_x = self.tcn_encoder_x(x1)
		encoded_x = torch.flatten(encoded_x,start_dim=1)
		encoded_x = torch.unsqueeze(encoded_x, dim=1)
		# print("enc flat",encoded_x.shape )
		encoded_trajectories_x.append(encoded_x)
		# print("context",context.shape)
		c1 = torch.transpose(context, 0, 2)
		# print("c1",c1.shape)
		encoded_context = self.context_conv(c1)
		# print("encoded_context 1",encoded_context.shape)

		encoded_context = self.relu(self.context_linear(encoded_context))
		# print("encoded_context 2",encoded_context.shape)
		
		appended_x = torch.cat((encoded_x,encoded_context),dim=2)
		# print("appended_x",appended_x.shape)

		encoded_appended_trajectories_x.append(appended_x)
		# print("encoded_context 2",len(encoded_appended_trajectories_x))
		
		# y1 = torch.transpose(y[:,:, agent][None, :, :], 1, 2)
		# encoded_y = self.tcn_encoder_y(y1)
		# encoded_y = torch.flatten(encoded_y)[None,None,:]
		# encoded_trajectories_y.append(encoded_y)

		# gat_input = torch.squeeze(torch.stack(encoded_appended_trajectories_x,dim=2))
	
		
		# if len(gat_input.shape) == 1:
		#     gat_input = torch.unsqueeze(gat_input,dim=0)

		# gat_output = self.gat(gat_input,adj)

		recon_y = []
		m = []
		var = []
		
		# pass all agents through decoder
		# for agent in range(x.shape[2]):
			
		# H_x = gat_output[agent].unsqueeze(0).unsqueeze(0)
		# H_xx = encoded_appended_trajectories_x[agent]
		# H_x = torch.cat((H_xx,H_x),dim=2)


		# H_y = encoded_trajectories_y[agent]
		# H_yy, means,log_var, z = self.cvae(H_y,H_x)
		H_yy =  torch.reshape(appended_x, (x.shape[2], 3, -1))
		# print('goal_vector',goal_vector.shape)
		# print('hyy', H_yy.shape)
		goal_expanded =self.goal_expand(goal_vector)
		goal_expanded=torch.reshape(goal_expanded.squeeze(0), (x.shape[2], -1))
		goal_expanded = torch.unsqueeze(goal_expanded, dim=1)

		# print("goalexp",goal_expanded.shape)
		full_encoded_x = torch.cat((appended_x.squeeze(1),goal_expanded.squeeze(1)),1)
		latent_space_appended.append( full_encoded_x)
		# print("full",full_encoded_x.shape)
		recon_y_x = (self.linear_decoder(H_yy))
		# print('recon_y_x1',recon_y_x.shape)

		# recon_y_x = torch.unsqueeze(recon_y_x,dim=0)
		# print('recon_y_x2',recon_y_x.shape,"x",x.shape)

		recon_y_x = acc_to_abs(recon_y_x,x)    
		# print('recon_y_x3',recon_y_x.shape)

		# recon_y.append(recon_y_x)
	
		# m.append(means)
		# var.append(log_var)
		return recon_y_x,full_encoded_x,var
	
	
	def inference(self,x,context,goal_vector):
	 

		encoded_trajectories_x = []
		encoded_appended_trajectories_x = []
		latent_space_appended = []
		
		# pass all agents through encoder
		# for agent in range(x.shape[2]):
		# 	x1 = torch.transpose(x[:,:, agent][None, :, :], 1, 2)
		# 	c1 = torch.transpose(context[:,:, agent][None, :, :], 1, 2)
		# 	encoded_context = self.context_conv(c1)
		# 	encoded_context = self.relu(self.context_linear(encoded_context))

		# 	encoded_x = self.tcn_encoder_x(x1)
		# 	encoded_x = torch.flatten(encoded_x)[None,None,:]
		# 	encoded_trajectories_x.append(encoded_x)
		# 	appended_x = torch.cat((encoded_x,encoded_context),dim=2)

		# 	encoded_appended_trajectories_x.append(appended_x)
		x1 = torch.transpose(x, 0, 2)
		encoded_x = self.tcn_encoder_x(x1)
		encoded_x = torch.flatten(encoded_x,start_dim=1)
		encoded_x = torch.unsqueeze(encoded_x, dim=1)
		encoded_trajectories_x.append(encoded_x)
		c1 = torch.transpose(context, 0, 2)
		encoded_context = self.context_conv(c1)
		encoded_context = self.relu(self.context_linear(encoded_context))
		appended_x = torch.cat((encoded_x,encoded_context),dim=2)

		encoded_appended_trajectories_x.append(appended_x)
		# gat_input = torch.squeeze(torch.stack(encoded_appended_trajectories_x,dim=2))
		
		# if len(gat_input.shape) == 1:
		#     gat_input = torch.unsqueeze(gat_input,dim=0)
		
		# gat_output = self.gat(gat_input,adj)
		
		recon_y = []
		m = []
		var = []
		
		# pass all agents through decoder
		# for agent in range(x.shape[2]):
		# 	# H_x = (gat_output[agent].unsqueeze(0)).unsqueeze(0)
		# 	# H_xx = encoded_appended_trajectories_x[agent]
		# 	# H_x = torch.cat((H_xx,H_x),dim=2)
		# 	# H_yy = self.cvae.inference(z,H_x)
		# 	H_yy =  torch.reshape(encoded_appended_trajectories_x[agent], (3, -1))

		# 	recon_y_x = (self.linear_decoder(H_yy)) 
		# 	recon_y_x = torch.unsqueeze(recon_y_x,dim=0)
		# 	recon_y_x = acc_to_abs(recon_y_x,x[:,:,agent][:,:,None])    

		# 	recon_y.append(recon_y_x.squeeze().detach())
		H_yy =  torch.reshape(appended_x, (x.shape[2], 3, -1))
		goal_expanded =self.goal_expand(goal_vector)
		goal_expanded=torch.reshape(goal_expanded.squeeze(0), (x.shape[2], -1))
		goal_expanded = torch.unsqueeze(goal_expanded, dim=1)
		full_encoded_x = torch.cat((appended_x.squeeze(1),goal_expanded.squeeze(1)),1)
		latent_space_appended.append( full_encoded_x)
		recon_y_x = (self.linear_decoder(H_yy))


		recon_y_x = acc_to_abs(recon_y_x,x)   
		return recon_y_x,full_encoded_x