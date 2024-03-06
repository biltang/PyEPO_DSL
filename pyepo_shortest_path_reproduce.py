import logging 
from pathlib import Path 
import sys 
import os 

import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch
import mlflow
from mlflow import MlflowClient
import mlflow 
import numpy as np
import pyepo 

# add system path to src directory
sys.path.append('/home1/yongpeng/PyEPO_DSL/')
from utils.experiment_utils import setup_mlflow_experiment, create_path_if_not_exist

sys.path.append('/home1/yongpeng/PyEPO_DSL/simulation_test/shortest_path_reproduce/')
from pyepo_shortest_path_model import LinearRegression, trainModel

# TODO: why does adding this line allow us to find pkg.pyepo.func.dsl.DSLoss?
sys.path.append('/home1/yongpeng/PyEPO_DSL/simulation_test/')
from simulation_test.dsl_icml_data import wrap_pyepo_dataset # TODO: why does adding this line allow us to find pkg.pyepo.func.dsl.DSLoss?

# settings
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
mlflow.set_tracking_uri("databricks")

@hydra.main(version_base=None, config_path="configs/pyepo_shortest_path_reproduce/", config_name="shortest_path_reproduce")
#@hydra.main(version_base=None, config_path="configs/icml/", config_name="icml_reproduce")
def main(cfg: DictConfig) -> None:
    
    logger.setLevel(level=cfg.general.log_level)
    
    # Print the entire configuration
    logger.debug("Config:\n%s", OmegaConf.to_yaml(cfg))
    
    # use gpus
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    if use_gpu==True:
        logger.info("Using GPU")               
    else:
        logger.info("Using CPU")
        
    # file path setup
    dataset_cfg = cfg.dataset
    path = Path(dataset_cfg.base_path) / dataset_cfg.setup / dataset_cfg.dataset_path # path to dataset directory
    logger.info(f"data path: {path}")
    
    # -------------------------------- 
    # MLflow setup 
    # --------------------------------
    
    # load mlflow experiment
    cur_data_instance_name = dataset_cfg.dataset_path
    experiment_name = f"{dataset_cfg.setup}_{cur_data_instance_name}" 
            
    experiment_name = setup_mlflow_experiment(client=MlflowClient(tracking_uri="databricks"),
                                              root_path=cfg.mlflow.root_path,
                                              experiment_name=experiment_name)
    mlflow.set_experiment(experiment_name)
    
    # set run name
    excluded_keys = ['loss_func'] # exclude these keys in constructing the run name
    # Initialize an empty list to collect the string components
    string_components = []

    for key, value in cfg.loss_func.items():
        # Check if the key is not in the list of excluded keys
        if key not in excluded_keys:
            # Construct the string component for this item
            component = f"{key}_{value}"
        
            # Append the component to the list
            string_components.append(component)

    for key, value in cfg.model.items():
        if key not in excluded_keys:
            # Construct the string component for this item
            component = f"{key}_{value}"
        
            # Append the component to the list
            string_components.append(component)
            
    # Join the components into a single string
    run_name = '-'.join(string_components)
    
    logging.info(f"Experiment name: {experiment_name} and run name: {run_name}")
    
    # set run parameters
    model_params = {**cfg.model, **cfg.loss_func}
    for e in excluded_keys:
        not_found = model_params.pop(e, 'Not Found')     
    # --------------------------------
    
    with mlflow.start_run(run_name=run_name) as run:
        # log params 
        mlflow.log_params(model_params)
        
        # set gpu tag
        mlflow.set_tag("gpu", use_gpu)
        
        optmodel = hydra.utils.instantiate(cfg.optmodel)
        
        loss = []
        val_regret = []
        regret = []
        for i in range(dataset_cfg.num_trials):
            # load test data 
            file = dataset_cfg.file + str(i) + dataset_cfg.file_extension
            file_path = path / file
            logger.info(f"data from {file_path} exists: {(file_path).exists()}")
   
            with open(str(file_path), 'rb') as file:
                dataset = pickle.load(file)
                    
            # construct dataset
            dataset_train = pyepo.data.dataset.optDataset(optmodel, 
                                                          dataset['train']['x'], 
                                                          dataset['train']['c'])
            dataset_val = pyepo.data.dataset.optDataset(optmodel, 
                                                          dataset['val']['x'], 
                                                          dataset['val']['c'])
            dataset_test = pyepo.data.dataset.optDataset(optmodel, 
                                                          dataset['test']['x'], 
                                                          dataset['test']['c'])
            
            loader_train = DataLoader(dataset_train, 
                                      batch_size=cfg.model.batch_size, 
                                      shuffle=True)
            loader_val = DataLoader(dataset_val,
                                     batch_size=cfg.model.batch_size,
                                     shuffle=False)
            loader_test = DataLoader(dataset_test,
                                     batch_size=cfg.model.batch_size,
                                     shuffle=False)
    
            with mlflow.start_run(run_name=run_name + '_trial_' + str(i), nested=True) as child_run:
                # initialize prediction model
                # 5 input features, 40 output features - hard coded from pyepo paper
                decision_aware_reg = LinearRegression(input_dim=5,
                                                      output_dim=40) 
                # instantiate loss function
                loss_func = hydra.utils.instantiate(cfg.loss_func.loss_func)
        
                if cfg.loss_func.loss_name == 'DSL':
                    h = len(dataset_train)**cfg.loss_func.h_exp
                    loss_func = loss_func(optmodel=optmodel,
                                          h=h,
                                          finite_diff_sch=cfg.loss_func.finite_diff_sch,
                                          processes=cfg.model.processes)
                else:
                    loss_func = loss_func(optmodel=optmodel,
                                          processes=cfg.model.processes)
                
                logger.debug(f"loss_func: {loss_func}")
                # train model
                loss_log, loss_log_regret = trainModel(reg=decision_aware_reg, 
                                                       loss_func=loss_func,
                                                       optmodel=optmodel,
                                                       loader_train=loader_train,
                                                       loader_test=loader_val,
                                                       use_gpu=use_gpu,
                                                       num_epochs=cfg.model.epochs,
                                                       lr=cfg.model.lr,
                                                       h_schedule=cfg.loss_func.h_sch,
                                                       lr_schedule=cfg.model.lr_schedule)
                
                # log test metrics
                test_regret = pyepo.metric.regret(decision_aware_reg, optmodel, loader_test)
                
                mlflow.log_metrics({"trial_train_loss": loss_log[-1]})
                mlflow.log_metrics({"trial_val_regret": loss_log_regret[-1]})
                mlflow.log_metrics({"trial_test_regret": test_regret})
                loss.append(loss_log[-1]) 
                val_regret.append(loss_log_regret[-1]) 
                regret.append(test_regret) 
                
            mlflow.log_metrics({"avg_train_loss": np.mean(loss)})
            mlflow.log_metrics({"avg_val_regret": np.mean(val_regret)})
            mlflow.log_metrics({"avg_test_regret": np.mean(regret)}) 
            
            
if __name__ == "__main__":
    main()