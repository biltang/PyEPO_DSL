import logging 
from pathlib import Path
import sys 
import time
import os 

from omegaconf import DictConfig, OmegaConf
import hydra
import pyepo 
from mlflow import MlflowClient
import pickle 

# settings
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# add system path to src directory
sys.path.append('/home1/yongpeng/PyEPO_DSL/')
from utils.experiment_utils import setup_mlflow_experiment, create_path_if_not_exist

def create_shortest_path_dataset(training_n: int=1000, num_feat: int=5, grid: tuple=(5,5), deg: int=2, e: float=0.1, sim_cfg: DictConfig={}, seed: dict={'train':1,'val':2,'test':3}):
    # generate training data
    x_train, c_train = pyepo.data.shortestpath.genData(training_n, 
                                                    num_feat, 
                                                    grid, 
                                                    deg, 
                                                    e,
                                                    seed=seed['train'])
    # generate validation data
    x_val, c_val = pyepo.data.shortestpath.genData(sim_cfg.val_n, 
                                                num_feat, 
                                                grid, 
                                                deg, 
                                                e,
                                                seed=seed['val'])
    
    # generate test data
    x_test, c_test = pyepo.data.shortestpath.genData(sim_cfg.test_n, 
                                                    num_feat, 
                                                    grid, 
                                                    deg, 
                                                    e,
                                                    seed=seed['test'])
    
    # get optDataset
    dataset_train = {"x": x_train, "c": c_train}
    dataset_val = {"x": x_val, "c": c_val}
    dataset_test = {"x": x_test, "c": c_test}
                    
    cur_dataset = {"train": dataset_train, "val": dataset_val, "test": dataset_test}
    
    return cur_dataset
    
    
@hydra.main(version_base=None, config_path="../../configs/pyepo_shortest_path_reproduce/", config_name="shortest_path_data")
def main(cfg: DictConfig):
    
    # instantiate credentials    
    os.environ['MLFLOW_TRACKING_USERNAME'] = cfg.dagshub.username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = cfg.dagshub.password

    sim_cfg = cfg.sim
    # Print the configuration
    logger.debug("Sim Config:\n%s", OmegaConf.to_yaml(sim_cfg))
    
    """Each problem instance is defined by
    1) number of training samples
    2) polynomial degree
    3) noise half-width
    
    We generate training, validation, and test data for each problem instance,
    and there are cfg.sim.num_trials instances in total.
    """
    
    for training_n in sim_cfg.training_n:
        
        for deg in sim_cfg.deg:
        
            for e in sim_cfg.noise_half_width:
                
                # create mlflow experiment and save data
                setup_name = sim_cfg.sim_setup_name
                cur_data_instance_name = f"n_{training_n}_deg_{deg}_e_{e}" 
                path = str(Path(__file__).parent.parent.parent / "data/simulation/" / setup_name / cur_data_instance_name) # relative path to other directory
    
                if cfg.mlflow.create_exp == True: # create mlflow experiment in this case
                    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
                    experiment_tags = {'sim_setup': setup_name,
                                       'n': str(training_n),
                                       'deg': str(deg), 
                                       'e': str(e)} 
                    logging.info(f"experiment_tags: {experiment_tags}")
                    experiment_id = setup_mlflow_experiment(client=client,
                                                root_path=cfg.mlflow.root_path,
                                                experiment_name=f"{setup_name}_{cur_data_instance_name}",
                                                experiment_tags=experiment_tags) 
    
                # create path
                create_path_if_not_exist(path)            
    
                logger.info(f"Generating data for {training_n} training samples with degree {deg} and noise half-width {e}")            
            
                # create datasets
                for i in range(sim_cfg.num_trials):
                    logger.info(f"Trial {i+1}")
                    
                    grid = (5,5) # grid size
                    num_feat = 5 # size of feature
                    start_seed = 3*i
                    seed = {'train': start_seed, 'val': start_seed+1, 'test': start_seed+2}
                    
                    cur_dataset = create_shortest_path_dataset(training_n=training_n, 
                                                               num_feat=num_feat,
                                                               grid=grid,
                                                               deg=deg,
                                                               e=e,
                                                               sim_cfg=sim_cfg,
                                                               seed=seed)
                    cur_dataset['seeds'] = seed
                    
                    # save data
                    with open(path + '/' + f'trial_{i}.pkl', 'wb') as file:
                        pickle.dump(cur_dataset, file)
                        
                        
if __name__ == "__main__":
    main()  