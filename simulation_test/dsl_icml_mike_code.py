import pdb 

import matplotlib.pyplot as plt
import numpy as np

def f1(x, m):
    return -((x <= 0.6) * (- 4 * x + 2) + (x > 0.6) * ((x - 0.6) * m - 0.4))

def f2(x, m, m0, c0):
    t_point = (c0 - 2)/m0
    return -((x <= t_point) * (m0 * x + 2) + (x > t_point) * ((x - t_point) * m + c0))

def generate_data_f1(n, m):
    x = np.random.uniform(0, 2, n)
    y = f1(x, m) + np.random.normal(0,.1,n)
    return x.reshape(-1,1), y.reshape(-1,1)

def generate_data_f2(n, m, m0, c0):
    x = np.random.uniform(0, 2, n)
    sd = .5
    # y = f2(x, m, m0, c0) + np.random.normal(0,.01,n)
    y = f2(x, m, m0, c0) + (np.random.exponential(sd,n) - sd)
    return x.reshape(-1,1), y.reshape(-1,1)


def plug(x, m, b):
    return (m * x + b)


def solve(pred):
    return (pred >= 0) * -1 + (pred < 0) * 1


def SPO_plus(pred, y_hat):
    alpha = 2
    t1 = -np.sum(solve(alpha * pred - y_hat) * (alpha * pred - y_hat))
    t2 = alpha * np.sum(solve(pred) * y_hat)
    t3 = np.sum(solve(pred) * pred)
    return (t1 + t2 - t3)/10


def V_func(pred):
    return np.sum(solve(pred) * pred)

    
def DSL(pred, y_hat, h):
    est_obj_1_s = V_func(pred + h * y_hat) 
    est_obj_2_s = - V_func(pred - h * y_hat)
    #pdb.set_trace()
    return (est_obj_1_s + est_obj_2_s)/(2*h)

def DSL_fwd(pred, y_hat, h):
    est_obj_1_s = V_func(pred + h * y_hat) 
    est_obj_2_s = - V_func(pred)
    return (est_obj_1_s + est_obj_2_s)/(h)

def DSL_bwd(pred, y_hat, h):
    est_obj_1_s = V_func(pred) 
    est_obj_2_s = - V_func(pred - h * y_hat)
    return (est_obj_1_s + est_obj_2_s)/(h)


def SPO_loss(b, y, x):
    pred = plug(x, b[0], b[1])
    return SPO_plus(pred, y)

def DSL_loss(b, y, x, h):
    pred = plug(x, b[0], b[1])
    return DSL(pred, y, h)

def DSL_fwd_loss(b, y, x, h):
    pred = plug(x, b[0], b[1])
    return DSL_fwd(pred, y, h)

def DSL_bwd_loss(b, y, x, h):
    pred = plug(x, b[0], b[1])
    return DSL_bwd(pred, y, h)

def ERM_loss(b, y, x):
    pred = plug(x, b[0], b[1])
    return np.mean(solve(pred) * y)
    