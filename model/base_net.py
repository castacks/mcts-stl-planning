import torch.nn.functional as F
from model.tcn import TemporalConvNet
import torch
from torch import nn
traj_no = 30

class Policy(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Policy, self).__init__()

        input_size = args.input_size
        num_channels = args.channel_size
        kernel_size = args.kernel_size
        dropout = args.dropout
        self.device = device

        self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.layer_1_nodes = 640
#         self.linear_encoder = nn.Linear(256,16)
        self.linear_decoder_x = nn.Linear(self.layer_1_nodes,256)
        self.linear_x = nn.Linear(256,traj_no)
        self.goal_expand = nn.Linear(10,128)

        self.context_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3,padding=1)
        self.context_linear = nn.Linear(15,7)
        self.relu2 = nn.ReLU()
        self.output_relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm1d(128)
        self.init_weights()

    def init_weights(self):
        self.linear_x.weight.data.normal_(0, 0.05)
        self.linear_decoder_x.weight.data.normal_(0, 0.05)
        self.goal_expand.weight.data.normal_(0, 0.05) 
        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)

    def forward(self, x1, goal_position):        

        # output = torch.zeros([x.shape[2],traj_no]).to(device)
        # softmax_out =torch.zeros([x.shape[2],traj_no]).to(device)
        # latent_space_ =torch.zeros([x.shape[2],self.layer_1_nodes]).to(device)
        
        encoded_trajectories_x = []
        encoded_appended_trajectories_x = []
        encoded_trajectories_y = []
        x1 = torch.unsqueeze(x1, 2)

        x1 = torch.transpose(x1, 0, 2)
#         print('x1',x1.shape)
#         if x.shape[2] <batch_size:
#             input_pos =torch.reshape(x1[:,:,0], (x.shape[2],3))
        # input_pos =torch.reshape(x1[:,:,0], (x.shape[2],3))
#         print("goal",goal_position.shape)
        if goal_position[0,0].cpu()<800 and goal_position[0,0].cpu()> -100 and abs(goal_position[0,1].cpu())<100: #1
            goal_position[0,0],goal_position[0,1] =0.0, 0.0

        elif goal_position[0,0].cpu()<1500 and goal_position[0,0].cpu()> 1000 and abs(goal_position[0,1].cpu())<100:  #2,
            goal_position[0,0],goal_position[0,1] =1450.0, 0.0
            
            
            
            
        # goal_vector = direction_detect(input_pos.cpu(), goal_position.cpu())
#         print('goal_res',goal_vector.shape)
        # goal_vector= goal_position.to(device)
        goal_expanded =self.goal_expand(goal_position)
#         print('goal_expand',goal_expanded.shape)
#         goal_position = (goal_position-mean)/std

#         x2 = torch.transpose(x1, 1, 2)

    
#         x2 = (x2[:,:,0:3]-mean[0:3])/std   #### convert this to km
        x1=x1/1000.0
#         x1 = torch.transpose(x2, 1, 2)
        encoded_x = self.tcn_encoder_x(x1)
#             print("fwd,encoded_x",agent, encoded_x.shape)
        encoded_x =encoded_x[:,:,-1]
        # print("mod,encoded_x", encoded_x.shape)
        full_encoded_x = torch.cat((encoded_x,goal_expanded),1)
#         print("full enc", full_encoded_x.shape)
#         print("full_encoded_x",full_encoded_x.shape)
        encoded_appended_trajectories_x.append(full_encoded_x)
        # pass all agents through encoder

        final_input = torch.squeeze(torch.stack(encoded_appended_trajectories_x))
#         print("final_input",final_input.shape)

        latent_space_ = final_input
            
            
        H_x = final_input
        recon_y_x = self.relu2(self.linear_decoder_x(H_x))
        recon_x = (self.linear_x(recon_y_x))
#         print("fwd recony",recon_y_x.shape,recon_x.shape)    

        output = torch.squeeze(recon_x,dim=0) ###### check
        # print(x1.shape)
    #     if int(x.shape[2]) == 1:
    #         softmax_out = F.softmax(recon_x, dim = 0)
    # #         print("softmax_out",softmax_out.shape)
    #     else:
        # print(recon_x.shape)   
        softmax_out = F.softmax(recon_x, dim = 0)
        return softmax_out, output    


class TCN(nn.Module):
    def __init__(self, args, device='cpu'):
        super(TCN, self).__init__()

        input_size = args.input_size
        num_channels = args.channel_size
        kernel_size = args.kernel_size
        dropout = args.dropout
        self.device = device
        self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.layer_1_nodes = 640
        self.linear_decoder_x = nn.Linear(self.layer_1_nodes, 256)
        self.linear_x = nn.Linear(256, traj_no)
        self.linear_decoder_v = nn.Linear(self.layer_1_nodes, 64)
        self.linear_v = nn.Linear(64, 64)
        self.output_v = nn.Linear(64, 1)
        self.goal_expand = nn.Linear(10, 128)

        self.context_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.context_linear = nn.Linear(15, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.output_relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.linear_x.weight.data.normal_(0, 0.05)
        self.linear_decoder_x.weight.data.normal_(0, 0.05)
        self.goal_expand.weight.data.normal_(0, 0.05)

        self.linear_decoder_v.weight.data.normal_(0, 0.05)
        self.linear_v.weight.data.normal_(0, 0.05)
        self.output_v.weight.data.normal_(0, 0.05)

        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)

    def forward(self, x1, goal):
        x1 = torch.unsqueeze(x1, 2)
        x1 = torch.transpose(x1, 0, 2)
        # goal_vector = goal.to(self.device)
        goal = goal.type(torch.float)
        # print("baseNet",goal)
        goal_expanded = self.goal_expand(goal)
        x1=x1/1000.0
        encoded_x = self.tcn_encoder_x(x1)

        encoded_x = encoded_x[:, :, -1]
        H_x = torch.cat((encoded_x, goal_expanded), 1)

        decoded1_x = self.relu((self.linear_decoder_x(H_x)))
        decoded2_x = self.linear_x(decoded1_x)

        decoded1_v = self.relu((self.linear_decoder_v(H_x)))
        decoded2_v = self.relu((self.linear_v(decoded1_v)))
        # multi_head = self.multi_head_layer(H_x)

        # output = torch.squeeze(decoded2,dim=0)

        softmax_out = F.softmax(decoded2_x, dim=1)
        # print(softmax_out.shape)

        v = self.tanh(self.output_v(decoded2_v))
        # print(v)
        return softmax_out[0], v
