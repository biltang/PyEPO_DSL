import pdb 

import numpy as np
import pyepo

def f_star(x: np.ndarray, m: float=0, m0: float=-4, c0: float=-0.2) -> np.ndarray:
    """generate expected y values from x values without noise

    Args:
        x (np.ndarray): one dimensional numpy array of x values
        m (float, optional): slope of segment x >= t_point. Defaults to 0.
        m0 (float, optional): default slope of segment x < t_point. Defaults to -4.
        c0 (float, optional): intersection of slopes m and m0 in output space y. Defaults to -0.2.

    Returns:
        np.ndarray: piecewise linear outcome values
    """
    t_point = (c0 - 2)/m0 # t_point is the x value of the intersection point of the two lines
    
    case_1 = (x < t_point) * (m0 * x + 2) # x is in region 1 with slope m0 (x < t_point)
    case_2 = (x >= t_point) * (m * (x - t_point) + c0) # x is in region 2 with slope m (x >= t_point)
    outcome = case_1 + case_2 # outcome is the sum of case_1 and case_2
    
    return outcome 

def gen_x_unif(n: int, x_min: float=0, x_max: float=2):
    """generate x values from uniform distribution

    Args:
        n (int): number of samples
        x_min (float): minimum value of x in uniform distribution
        x_max (float): maximum value of x in uniform distribution

    Returns:
        numpy vector: n samples from uniform distribution
    """
    return np.random.uniform(x_min, x_max, n)

def Y(exp_Y: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """get noisy y vector

    Args:
        exp_Y (np.ndarray): exp y values
        noise (np.ndarray): noise vector

    Returns:
        np.ndarray: noisy y vector
    """
    return exp_Y + noise
    
def gen_noise(n: int, alpha: float=1, expon_offset: float=0.5, expon_rate: float=0.5, norm_std: float=0.25) -> np.ndarray:
    """generate noise according to icml dsl paper

    Args:
        n (int): number of samples
        alpha (float, optional): weighting between exponential vs gaussian noise. Defaults to 1.
        expon_offset (float, optional): offset to exponential noise component. Defaults to 0.5 from paper.
        expon_rate (float, optional): rate parameter for exponential rv. Defaults to 2 so that expon rv has mean 0.5 from paper.
        norm_std (float, optional): std dev for gaussian noise component. Defaults to 0.25 from paper.

    Returns:
        np.ndarray: array of noise values
    """
    
    expon_noise_comp = np.random.exponential(expon_rate, n) - expon_offset
    gamma_noise_comp = np.random.normal(loc=0, scale=norm_std, size=n)
    
    noise = (np.sqrt(alpha) * expon_noise_comp) + (np.sqrt(1 - alpha) * gamma_noise_comp)
    return noise

def gen_data_instance(n: float, x_config: dict={}, exp_y_config: dict={}, noise_config: dict={}):
    """generate a single data instance

    Args:
        n (float): number of samples
        x_config (dict): configuration for x values
        exp_y_config (dict): configuration for expected y values
        noise_config (dict): configuration for noise

    Returns:
        tuple: x, exp_y, noise, y
    """
    x = gen_x_unif(n, **x_config).reshape(-1,1)
    exp_y = f_star(x, **exp_y_config).reshape(-1,1)
    noise = gen_noise(n, **noise_config).reshape(-1,1)
    y = Y(exp_y, noise).reshape(-1,1)
    return x, exp_y, noise, y

def wrap_pyepo_dataset(optmodel, x, costs):
    return pyepo.data.dataset.optDataset(optmodel, x, costs)    