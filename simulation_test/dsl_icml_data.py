from dataclasses import dataclass, field
import sys 
import pdb 
from pathlib import Path
import logging 

import numpy as np
import pyepo
import hydra 
from mlflow import MlflowClient
from hydra.core.config_store import ConfigStore
import pickle 

# add system path to src directory
sys.path.append('/home1/yongpeng/PyEPO_DSL/')
from utils.experiment_utils import setup_mlflow_experiment, create_path_if_not_exist

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


# ----------------------------------------------------------
# Programmatically generate data and mlflow experiments
# -----------------------------------------------------------
""" 
The following code is used to programmatically generate data and mlflow experiments.
It can be safely ignored if one wants to just generate data using the functions
from above.
"""
@dataclass
class XConfig:
    # x config values
    x_min: float = 0 # minimum value of x in uniform distribution
    x_max: float = 2 # maximum value of x in uniform distribution
    
@dataclass
class ExpYConfig:
    # Expected y config values
    m: float = 0 # slope of segment x >= t_point
    m0: float = -4 # default slope of segment x < t_point
    c0: float = -0.2 # intersection of slopes m and m0 in output space y
    noisy_y_save: bool = False # save noisy y values
    
@dataclass
class NoiseConfig:
    # Noise config values
    alpha: float = 1 # weighting between exponential vs gaussian noise
    expon_offset: float = 0.5 # offset to exponential noise component
    expon_rate: float = 2 # rate parameter for exponential rv
    norm_std: float = 0.25 # std dev for gaussian noise component

@dataclass
class MLFlowConfig:
    create_exp: bool = False
    root_path: str = "/Users/yongpeng@usc.edu/"
    
@ dataclass
class SaveInfo:
    sim_setup_name: str = "icml_simulation"
    save_name: str = "data.pkl"
    
@dataclass
class GenDataConfig:
    # optimization model
    optmodel: str 
    
    n: int = 10000 # number of samples
    
    x_config: XConfig = field(default_factory=XConfig) # x config values
    y_config: ExpYConfig = field(default_factory=ExpYConfig) # Expected y config values
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    
    # mlflow config
    mlflow: MLFlowConfig = field(default_factory=MLFlowConfig)
    
    # save info
    save_info: SaveInfo = field(default_factory=SaveInfo)
    
    
cs = ConfigStore.instance()
cs.store(name="icml_gen_data_cfg", node=GenDataConfig)

def wrap_pyepo_dataset(cfg: GenDataConfig):
    # ---------------------------
    # Generate PyEPO Dataset
    # ---------------------------
    x, exp_y, noise, y = gen_data_instance(n=cfg.n,
                                           x_config=cfg.x_config,
                                           exp_y_config=cfg.y_config,
                                           noise_config=cfg.noise_config)
    optmodel = hydra.utils.instantiate(cfg.optmodel)
    
    if cfg.save_info.noisy_y_save == True:
        dataset = pyepo.data.dataset.optDataset(optmodel, x, y)
    else:
        dataset = pyepo.data.dataset.optDataset(optmodel, x, exp_y)
        
    return dataset


@hydra.main(version_base=None, config_path="../configs/icml/sim/", config_name="gen_data")
def main(cfg: GenDataConfig):
    
    # ---------------------------
    # Generate PyEPO Dataset
    # ---------------------------
    dataset = wrap_pyepo_dataset(cfg)
    
    # ---------------------------
    # Save Data
    # ---------------------------
    """Simulation data instance/experiment is configured by:
    1) slopes m of the piecewise linear function
    """
    # create mlflow experiment and save data
    setup_name = cfg.save_info.sim_setup_name
    
    cur_data_instance_name = f"slope-m-{cfg.y_config.m}" 
    path = str(Path(__file__).parent.parent / "data/simulation/" / setup_name / cur_data_instance_name) # relative path to other directory
    
    """By default, config file config.mlflow.create_exp is set to False, and below block
    will not execute
    """
    if cfg.mlflow.create_exp == True: # create mlflow experiment in this case
        client = MlflowClient(tracking_uri="databricks")
        experiment_tags = {'sim_setup': cfg.save_info.sim_setup_name,
                           'slope_m': str(cfg.y_config.m),                           
                           'n': str(cfg.n)} 
        logging.info(f"experiment_tags: {experiment_tags}")
        experiment_id = setup_mlflow_experiment(client=client,
                                                root_path=cfg.mlflow.root_path,
                                                experiment_name=f"{cfg.save_info.sim_setup_name}_{cur_data_instance_name}",
                                                experiment_tags=experiment_tags) 
    
    # create path
    create_path_if_not_exist(path)                    
    
    # save data
    with open(path + '/' + cfg.save_info.save_name, 'wb') as file:
        pickle.dump(dataset, file)
        
        
if __name__ == "__main__":
    main()  