import time
import pdb 

import numpy as np
import gurobipy as gp
from gurobipy import GRB    
from torch import nn
import torch 
from pyepo.model.opt import optModel
from pyepo.model.grb import optGrbModel
from pyepo import EPO 
import pyepo 

# build linear model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    
# build optimization model
class icmlCondLinOptModel(optModel):
    def __init__(self):
        self.modelSense = EPO.MAXIMIZE
        self.z = np.array([0])
        
        super().__init__()
        
    def _getModel(self):
        model = {'cost': None, 'z': self.z}    
        return model, model['z']
    
    def setObj(self,c):
        self._model['cost'] = c[0]
    
    def solve(self):
        
        if self._model['cost'] >= 0:
            self.z[0] = 1
        else:
            self.z[0] = -1 
        
        obj = self._model['cost']*self.z 
        
        return self.z, obj 

# gurobi optimization model
class icmlCondLinOptModel_gurobi(optGrbModel):
    def __init__(self):
        self.modelSense = EPO.MAXIMIZE 
        super().__init__()
        
    def _getModel(self):
        m = gp.Model('cond_lin_opt')
        
        # variables
        z = m.addVars(1, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='z')
        
        # sense 
        m.modelSense = GRB.MAXIMIZE
        
        return m, z 
    
    
# train model
def trainModel(reg, loss_func, optmodel, loader_train, loader_test, num_epochs=20, lr=1e-2):
    # set adam optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train mode
    reg.train()
    # init log
    loss_log = []
    loss_log_regret = [pyepo.metric.regret(reg, optmodel, loader_test)]
    # init elpased time
    elapsed = 0
    for epoch in range(num_epochs):
        # start timing
        tick = time.time()
        # load data
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            # cuda
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            cp = reg(x)
            loss = loss_func(cp, c, w, z)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record time
            tock = time.time()
            elapsed += tock - tick
            # log
            loss_log.append(loss.item())
        regret = pyepo.metric.regret(reg, optmodel, loader_test)
        loss_log_regret.append(regret)
        print("Epoch {:2},  Loss: {:9.4f},  Regret: {:7.4f}%".format(epoch+1, loss.item(), regret*100))
    print("Total Elapsed Time: {:.2f} Sec.".format(elapsed))
    return loss_log, loss_log_regret