from numpy.lib.function_base import disp
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model.trajairnet_tcn import TrajAirNet
from model.trajairnet_tcn_goalgail import TrajAirNetGoalGAIL
import numpy as np
import torch
import torch.optim as optim

class Net():

    def __init__(self,args):
        
        self.args = args
        if args.algo == "BC":
            self.nnet = TrajAirNet(args)
        elif args.algo == "GAIL":
            self.nnet = TrajAirNetGoalGAIL(args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)

        if args.model_weights is not None:
            modelpath = self.args.base_path + args.models_folder + args.model_weights
            self.load_hot_start(modelpath)

    def load_hot_start(self, modelpath):
        print(" Loading model weights from ", modelpath)

        checkpoint = torch.load(modelpath, map_location=torch.device('cpu'))
        miss, unex = self.nnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # print("miss", miss, "unex", unex)
        print(" Pre-trainined model weights loaded from ", modelpath)

    def train(self,examples):
        print("Device is ",self.device)
        # self.nnet.to(self.device)
        self.nnet.train()
        optimizer = optim.AdamW(self.nnet.parameters(), lr =0.001)

        target = np.hstack([x[3] for x in examples])
        print('target train -1/1: {}/{}'.format(len(np.where(target == -1)[0]), len(np.where(target == 1)[0])))
        idx = np.where(target == 1)[0]
        pos_samples = [examples[i] for i in idx] 
        idx = np.where(target == -1)[0]
        neg_samples = [examples[i] for i in idx] 
        if self.args.balance_data and len(neg_samples) != 0 and len(pos_samples) != 0 :

            rep_count = int(np.ceil(len(neg_samples)/len(pos_samples)))

            pos_samples = pos_samples*rep_count
            examples = pos_samples + neg_samples
            print("Balanced training samples are ", len(examples))

        train_data = ReplayDataLoader(examples)

      
        train_dataloader = DataLoader(train_data, batch_size=64,shuffle=True)


        for epoch in range(self.args.epochs):
            loss_pi = 0
            loss_v = 0

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()

                batch = [tensor.to(self.device) for tensor in batch]
                position,goal, target_pis, target_vs = batch
                total_loss = 0
                for i in range(position.shape[0]):
                    # print(target_vs[i])
                    out_pi, out_v = self.nnet(position[i],goal[i])
                    l_pi = self.loss_pi(target_pis[i], out_pi)
                    l_v = self.loss_v(target_vs[i], out_v)
                    total_loss += l_pi + l_v
                    loss_v += l_v.item()  
                    loss_pi += l_pi.item()
                    assert(not np.isnan(loss_v))
                    assert(not np.isnan(loss_pi))

                total_loss.backward()
                optimizer.step()
            print("Epoch #",epoch,"Loss_pi", loss_pi, "Loss_v", loss_v)
        
    def predict(self,curr_position,goal_position):
        # print("ha",curr_position.shape)
        # print(curr_position[9:,:])
        # self.nnet.eval()
        context = torch.zeros((11,2,1), device = self.device)
        # self.nnet.to('cpu')
        
        if self.args.algo == "BC":
            pred = self.nnet.inference(torch.unsqueeze(curr_position[9:,:],2), context)[0]
        elif self.args.algo == "GAIL":
            pred = self.nnet.inference(torch.unsqueeze(curr_position[9:,:],2), context,goal_position)[0][0]

        # print("he",np.linalg.norm(curr_position[-1,:]-pred[:,0]))
        # print("he",np.linalg.norm(pred[:,0]-pred[:,1]))

        # print("ha",pred.shape)
        return pred.detach().cpu()
    
    
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * (outputs)) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return (targets - outputs.view(-1)) ** 2


class ReplayDataLoader(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def balance_data(self):

        pass
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        
        position = self.examples[index][0]
        goal = self.examples[index][1]
        pi = self.examples[index][2]
        v = self.examples[index][3]

        return position,goal, pi, v  
