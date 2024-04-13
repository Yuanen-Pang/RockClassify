import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import os
join = os.path.join

from model import LinearClassifier
from dataset import RockData
from pyn.Ptorch.exp import DLAgent
from pyn.nmp import shuffle_same
from pyn import Json

class MyAgent(DLAgent):
    def __init__(self, args, continue_train=False, debug=False) -> None:
        super().__init__(args, continue_train, debug)
        self.args = super().get_args()
        X_train, X_test = np.load(self.args.data['X_train']), np.load(self.args.data['X_test'])
        y_train, y_test = np.load(self.args.data['y_train']), np.load(self.args.data['y_test'])
        # X_train = X_train[:int(self.args.data['ratio']*len(X_train))]
        # y_train = y_train[:int(self.args.data['ratio']*len(y_train))]
        # print(len(X_train))
        # X_train = X_train[:self.args.data['num']]
        # y_train = y_train[:self.args.data['num']]
        # y_train[(y_train > -1) & (y_train <= 4)] = 0
        # y_train[(y_train > 4) & (y_train <= 7)] = 1
        # y_train[(y_train > 7) & (y_train <= 14)] = 2
        # y_test[(y_test > -1) & (y_test <= 4)] = 0
        # y_test[(y_test > 4) & (y_test <= 7)] = 1
        # y_test[(y_test > 7) & (y_test <= 14)] = 2
        
        train_dataset, test_dataset = \
            RockData(X_train, y_train, train=True), RockData(X_test, y_test, train=False)
        
        batch_size = self.args.train['batch_size']
        self.train_dataloader = DataLoader(train_dataset, batch_size, True, num_workers=16)
        self.dev_dataloader = DataLoader(test_dataset, batch_size, False, drop_last=False, num_workers=16)
        
    def get_opt(self):
        params = self.model.parameters()
        lr = self.args.train['lr']
        if self.args.train['opt'][0] == 'adam':
            optimizer = optim.Adam(params,lr)
        elif self.args.train['opt'][0] == 'sgd':
            optimizer = optim.SGD(params,lr,momentum=self.args.train['opt'][1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.args.train['scheduler'], 0.1)
        weight=torch.tensor([1/2729, 1/4683, 1/5188]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        return optimizer, criterion, scheduler
    
    def get_model(self):
        name = self.args.model['name']
        model_dict = {'linear': LinearClassifier}
        return model_dict[name](**self.args.model[name])
    
    def get_data(self, flag):
        if flag == 'train':
            return self.train_dataloader
        else:
            return self.dev_dataloader
    
    def cal_train_loss(self, train_loader, optimizer, criterion, scheduler):
        self.model.train()
        total_loss = []
        class_pred = []
        class_true = []
        for X, Y in tqdm(train_loader):
            X, Y = X.to(self.device), Y.to(self.device)
            optimizer.zero_grad()
            Y_hat = self.model(X)
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            class_pred.extend(Y_hat.argmax(1))
            class_true.extend(Y)
        acc = sum(torch.tensor(class_pred)==torch.tensor(class_true))/len(class_pred)
        scheduler.step() if scheduler else None
        return {'train_loss':sum(total_loss)/len(total_loss), "train_acc":float(acc)}

    def cal_dev_loss(self, dev_loader, criterion):
        self.model.eval()
        total_loss = []
        class_pred = []
        class_true = []
        with torch.no_grad():
            for X, Y in tqdm(dev_loader):
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat = self.model(X)
                loss = criterion(Y_hat, Y)
                total_loss.append(loss.item())
                class_pred.extend(Y_hat.argmax(1))
                class_true.extend(Y)
        acc = sum(torch.tensor(class_pred)==torch.tensor(class_true))/len(class_pred)
        return {'dev_loss':sum(total_loss)/len(total_loss), "dev_acc":float(acc)}
    
    def test(self):
        self.model.eval()
        self.load_model()
        dev_loader = self.get_data(flag='test')
        class_pred = []
        class_true = []
        # probs = []
        with torch.no_grad():
            for X, Y in tqdm(dev_loader):
                X, Y = X.to(self.device).float(), Y.to(self.device).long().to(self.device)
                Y_hat = self.model(X)
                class_pred.extend(Y_hat.argmax(1).cpu())
                class_true.extend(Y.cpu())
                # probs.extend(prob.cpu().detach().numpy())
            np.save(join(self.args.save['root'], self.args.save['result_path']) ,np.concatenate([[class_pred],[class_true]],axis=0))
        
    def test_prob(self):
        self.model.eval()
        self.load_model()
        dev_loader = self.get_data(flag='test')
        class_pred = []
        # probs = []
        with torch.no_grad():
            for X, Y in tqdm(dev_loader):
                X, Y = X.to(self.device).float(), Y.to(self.device).long().to(self.device)
                Y_hat = self.model(X)
                class_pred.append(Y_hat.softmax(dim=1).cpu())
                # probs.extend(prob.cpu().detach().numpy())
            np.save(join(self.args.save['root'], self.args.save['prob_path']) ,np.concatenate(class_pred,axis=0))
