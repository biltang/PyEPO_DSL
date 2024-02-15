import numpy as np
import torch


from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass, _cache_in_pass

class DSLoss(optModule):
    """
    An autograd module wrapper for DS Loss (DSLFunc) 

    Args:
        optModule (_type_): _description_
    """
    def __init__(self, optmodel, h=1, finite_diff_sch='B', processes=1, solve_ratio=1, dataset=None):
        """_summary_

        Args:
            optmodel (_type_): _description_
            h (int, optional): _description_. Defaults to 1.
            finite_diff_sch (str, optional): _description_. Defaults to 'B'.
            processes (int, optional): _description_. Defaults to 1.
            solve_ratio (int, optional): _description_. Defaults to 1.
            dataset (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        ### the finite difference step size h must be positive
        if h < 0:
            raise ValueError("h must be positive")
        
        super().__init__(optmodel, processes, solve_ratio, dataset)
    
        self.dsl = DSLFunc()
        self.h = h
        self.finite_diff_sch = finite_diff_sch

    def forward(self, pred_cost, true_cost, true_sol, true_obj, reduction='mean'):
        """_summary_

        Args:
            pred_cost (_type_): _description_
            true_cost (_type_): _description_
            true_sol (_type_): _description_
            true_obj (_type_): _description_
            reduction (str, optional): _description_. Defaults to 'mean'.
        """
        loss = self.dsl.apply(pred_cost, true_cost, true_sol, true_obj,
                              self.optmodel, self.processes, self.pool,
                              self.h, self.finite_diff_sch)
    
        #print(loss.shape)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss

class DSLFunc(torch.autograd.Function):
    """_summary_

    Args:
        torch (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    @staticmethod
    def forward(ctx, 
                pred_cost, true_cost, true_sol, true_obj,
                optmodel, processes, pool,
                h, finite_diff_sch='B'):
        """forward pass to calculate DS loss

        Args:
            ctx (_type_): pytorch context manager
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            h (float): step size to use in finite difference approximation
            finite_diff_sch (str, optional): Specify type of finite-difference scheme, Backward Differencing/PGB ('B')
                                             or Central Differencing/PGC ('C'). Defaults to 'B'.

        Returns:
            torch.tensor: DSL loss
        """
        
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        c = true_cost.detach().to("cpu").numpy()
        w = true_sol.detach().to("cpu").numpy() # QUESTION: do we need w in this function?
        z = true_obj.detach().to("cpu").numpy() # QUESTION: do we need z in this function?
        
        # Set the plug-in cp for the plus term in finite differencing depending on PGB vs PGC
        # also modify step_size h used in the calculation of finite difference loss function
        if finite_diff_sch == 'C':
            cp_plus = cp + h*c
            step_size = 1/(2*h)
        else:
            cp_plus = cp
            step_size = 1/h
        
        # plug-in cp for minus term in finite differencing
        cp_minus = cp - h*c 
        
        #### TODO: SOLUTION POOLING/CACHE####
        sol_plus, obj_plus = _solve_in_pass(cp=cp_plus,
                                            optmodel=optmodel,
                                            processes=processes,
                                            pool=pool)
        
        sol_minus, obj_minus = _solve_in_pass(cp=cp_minus,
                                              optmodel=optmodel,
                                              processes=processes,
                                              pool=pool)
        
        # Calculate loss - a function of cp, c, h
        loss = step_size * (obj_plus - obj_minus)
        
        # sense
        if optmodel.modelSense == EPO.MINIMIZE:
            loss = np.array(loss)
        if optmodel.modelSense == EPO.MAXIMIZE:
            loss = - np.array(loss)
    
        #print(loss)
        
        # Convert numpy vectors back to tensors
        loss = torch.FloatTensor(loss).to(device)
        loss = loss.unsqueeze(1) 
        
        sol_plus = torch.FloatTensor(np.array(sol_plus)).to(device) # QUESITON: do we need the np.array conversion of sol_plus from _solve_in_pass()?
        sol_minus = torch.FloatTensor(np.array(sol_minus)).to(device) # QUESITON: do we need the np.array conversion of sol_minus from _solve_in_pass()?
        
        #print(loss.shape)
        # To calculate gradient in backwards pass for DSL, need to save the sol_plus, sol_minus terms
        # save solutions for backwards pass
        ctx.save_for_backward(sol_plus, sol_minus)
        
        # save the optimization model so we can get the model sense in backwards pass
        ctx.optmodel = optmodel 
        ctx.step_size = step_size
        
        return loss
        
    @staticmethod
    def backward(ctx, grad_output):
        """_summary_

        Args:
            ctx (_type_): _description_
            grad_output (_type_): _description_

        Returns:
            _type_: _description_
        """
        sol_plus, sol_minus = ctx.saved_tensors
        optmodel = ctx.optmodel 
        step_size = ctx.step_size
        
        # gradient of dsl loss with respect to predicted costs vector cp - d_l/d_cp
        grad = step_size * (sol_plus - sol_minus)
        
        # if maximization, reverse direction of gradient
        if optmodel.modelSense == EPO.MAXIMIZE:
            grad = -1 * grad 
        
        #print(grad.shape)
        #print(grad_output.shape)
        return grad_output*grad, None, None, None, None, None, None, None, None 