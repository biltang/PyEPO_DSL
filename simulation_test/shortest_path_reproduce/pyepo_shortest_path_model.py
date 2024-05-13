# build optModel
import time
import os 

from pyepo.model.grb import optGrbModel
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch 
import pyepo 
import numpy as np


class shortestPathModel(optGrbModel):

    def __init__(self):
        self.grid = (5,5)
        self.arcs = self._getArcs()
        super().__init__()

    def _getArcs(self):
        """
        A helper method to get list of arcs for grid network

        Returns:
            list: arcs
        """
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        import gurobipy as gp
        from gurobipy import GRB
        # ceate a model
        m = gp.Model("shortest path")
        # varibles
        x = m.addVars(self.arcs, name="x")
        # sense
        m.modelSense = GRB.MINIMIZE
        # flow conservation constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= x[e]
                # source
                if i == 0 and j == 0:
                    m.addConstr(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                    m.addConstr(expr == 1)
                # transition
                else:
                    m.addConstr(expr == 0)
        return m, x
    

# build linear model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0.0001, path='./checkpoints/pyepo_shortest_path/checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to. 
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_regret = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_regret, model):
        
        if self.best_regret is None:
            self.best_regret = val_regret
            self.save_checkpoint(val_regret, model)
            
        elif val_regret > self.best_regret - self.delta:
            
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.save_checkpoint(val_regret, model)
            self.best_regret = val_regret
            self.counter = 0

    def save_checkpoint(self, val_regret, model):
        """Saves model when validation loss decrease."""
        # Check if the checkpoints directory exists, otherwise create it
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
            
        if self.verbose:
            print(f'Validation regret decreased ({self.best_regret:.6f} --> {val_regret:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        
        
# train model
def trainModel(reg, loss_func, loss_name, optmodel, loader_train, loader_test, use_gpu=False, num_epochs=20, lr=1e-2, h_schedule=False, lr_schedule=False, early_stopping_cfg=None):
    # set adam optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    
    if lr_schedule == True:
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1) # Define scheduler
    
    if early_stopping_cfg is not None and early_stopping_cfg['enabled']:
        print("Early Stopping Enabled")
        early_stopping = EarlyStopping(patience=early_stopping_cfg['patience'], 
                                       verbose=True, 
                                       delta=early_stopping_cfg['delta'], 
                                       path=early_stopping_cfg['checkpoint_path'])
    
        
    # train mode
    reg.train()
    # init log
    loss_log = []
    train_regret_log = [pyepo.metric.regret(reg, optmodel, loader_train)]
    val_regret_log = [pyepo.metric.regret(reg, optmodel, loader_test)]
    # init elpased time
    start = time.time()
    for epoch in range(num_epochs):
        # start timing
        
        if epoch%10 == 0 and h_schedule == True:
            loss_func.h = loss_func.h/2
            print("h: ", loss_func.h)
            
        # load data
        batch_loss = [0]
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            # cuda
            if use_gpu == True:
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            cp = reg(x)
            
            if loss_name in ['MSE', 'CosineSurrogate', 'CosineMSE']:
                loss = loss_func(cp, c)
            elif loss_name == 'Cosine':
                target = torch.ones(c.size(0))
                if use_gpu == True:
                    target = target.cuda()
                loss = loss_func(cp, c, target=target)
            else:
                loss = loss_func(cp, c, w, z)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record time
            #tock = time.time()
            #elapsed += tock - tick
            # log
            batch_loss.append(loss.item())
        
        loss_log.append(np.mean(batch_loss))
            
        if lr_schedule == True:
            scheduler.step()
        
        # train regret
        train_regret = pyepo.metric.regret(reg, optmodel, loader_train)
        train_regret_log.append(train_regret)
        
        # val regret
        val_regret = pyepo.metric.regret(reg, optmodel, loader_test)
        val_regret_log.append(val_regret)
        print("Epoch {:2},  Loss: {:9.4f},  Train_Regret: {:7.4f}%, Val_Regret: {:7.4f}%".format(epoch+1, loss_log[-1],  train_regret*100, val_regret*100))
        
        if early_stopping_cfg is not None and early_stopping_cfg['enabled']:
            # early stopping
            early_stopping(val_regret, reg)
        
            if early_stopping.early_stop:
                print("We are at epoch:", epoch+1 - early_stopping.patience)
                break
        
    end = time.time()
    print("Total Elapsed Time: {:.2f} Sec.".format(end-start))
    
    if early_stopping_cfg is not None and early_stopping_cfg['enabled']:
        load_epoch = epoch + 1 - early_stopping.patience
        return loss_log[load_epoch-1], train_regret_log[load_epoch], val_regret_log[load_epoch]
    else:
        return loss_log[-1], train_regret_log[-1], val_regret_log[-1]