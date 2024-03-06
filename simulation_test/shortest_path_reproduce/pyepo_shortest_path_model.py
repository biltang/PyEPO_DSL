# build optModel
import time

from pyepo.model.grb import optGrbModel
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch 
import pyepo 

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
    
# train model
def trainModel(reg, loss_func, optmodel, loader_train, loader_test, use_gpu=False, num_epochs=20, lr=1e-2, h_schedule=False, lr_schedule=False):
    # set adam optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    
    if lr_schedule == True:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # Define scheduler
    
    # train mode
    reg.train()
    # init log
    loss_log = []
    loss_log_regret = [pyepo.metric.regret(reg, optmodel, loader_test)]
    # init elpased time
    start = time.time()
    for epoch in range(num_epochs):
        # start timing
        
        if epoch%10 == 0 and h_schedule == True:
            loss_func.h = loss_func.h/2
            print("h: ", loss_func.h)
            
        # load data
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            # cuda
            if use_gpu == True:
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            cp = reg(x)
            loss = loss_func(cp, c, w, z)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record time
            #tock = time.time()
            #elapsed += tock - tick
            # log
            loss_log.append(loss.item())
            
        if lr_schedule == True:
            scheduler.step()
            
        regret = pyepo.metric.regret(reg, optmodel, loader_test)
        loss_log_regret.append(regret)
        print("Epoch {:2},  Loss: {:9.4f},  Regret: {:7.4f}%".format(epoch+1, loss.item(), regret*100))
    end = time.time()
    print("Total Elapsed Time: {:.2f} Sec.".format(end-start))
    return loss_log, loss_log_regret