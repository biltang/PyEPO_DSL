import logging 
from pathlib import Path
import time 
import sys 
import pdb 
import os 

import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch
from sklearn.linear_model import LinearRegression as sk_lr
from mlflow import MlflowClient
import mlflow 
import numpy as np
import pyepo 


# add system path to src directory
sys.path.append('/home1/yongpeng/PyEPO_DSL/')
sys.path.append('/home1/yongpeng/PyEPO_DSL/simulation_test/')
from utils.experiment_utils import setup_mlflow_experiment, create_path_if_not_exist
from simulation_test.dsl_icml_data import wrap_pyepo_dataset
from simulation_test.dsl_icml_model import LinearRegression, trainModel


# settings
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

mlflow.set_tracking_uri("databricks")


@hydra.main(version_base=None, config_path="configs/icml/", config_name="icml_reproduce")
def main(cfg: DictConfig) -> None:

    # use gpus
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    
    dataset_cfg = cfg.dataset
    path = Path(dataset_cfg.base_path) / dataset_cfg.setup / dataset_cfg.dataset_path # path to dataset directory
    test_file = dataset_cfg.test + dataset_cfg.file_extension # test file name
    logger.info(f"data path: {path}")
    
    # load test data 
    logger.info(f"test_data from {path / test_file} exists: {(path / test_file).exists()}")
    with open(str(path / test_file), 'rb') as file:
        dataset_test = pickle.load(file)
    
    loader_test = DataLoader(dataset_test, cfg.model.batch_size, shuffle=False)
    
    # -------------------------------- 
    # MLflow setup 
    # --------------------------------
    
    # load mlflow experiment
    cur_data_instance_name = f"slope-m-{cfg.sim.y_config.m}"
    experiment_name = f"{cfg.sim.save_info.sim_setup_name}_{cur_data_instance_name}" 
            
    experiment_name = setup_mlflow_experiment(client=MlflowClient(tracking_uri="databricks"),
                                              root_path=cfg.mlflow.root_path,
                                              experiment_name=experiment_name)
    mlflow.set_experiment(experiment_name)
    
    run_name = cfg.model.model_name + f"_n-{cfg.sim.n}_epochs-{cfg.model.epochs}_gpu-{use_gpu}"
    model_params = {'model_name': cfg.model.model_name,
                    'epochs': cfg.model.epochs,
                    'gpu': use_gpu,
                    'n': cfg.sim.n} 
    
    if cfg.model.model_name == 'DSL':
        run_name = run_name + f"_h-{cfg.model.h_exp}_finite-diff-sch-{cfg.model.finite_diff_sch}"

        # important model params to log
        additional_params = {'h': cfg.model.h_exp,
                             'finite_diff_sch': cfg.model.finite_diff_sch}
        model_params = {**model_params, **additional_params}
    elif cfg.model.model_name == 'MSE':
        run_name = f'MSE_n-{cfg.sim.n}'
        model_params = {'model_name': 'MSE',
                         'n': cfg.sim.n}
        
    # --------------------------------
    
    with mlflow.start_run(run_name=run_name) as run:
        # log params 
        mlflow.log_params(model_params)
        
        # set gpu tag
        mlflow.set_tag("gpu", use_gpu)
        
        regret = []
        min_regret = []
        for i in range(cfg.num_trials):
            
            logger.info(f"Trial {i}")
            
            # generate training data
            dataset_train = wrap_pyepo_dataset(cfg.sim)
            loader_train = DataLoader(dataset_train, 
                                        batch_size=cfg.model.batch_size, 
                                        shuffle=True)
        
            opt_model = dataset_train.model

            # nominal predictor - for weight initialization
            x_train, c_train, w_train, z_train = dataset_train[:]
            
            reg = sk_lr() # regular linear regression model for initializing weights
            reg = reg.fit(x_train, c_train) 
            logger.info(f"init coef: {reg.coef_} and init intercept: {reg.intercept_}")
            
            with mlflow.start_run(run_name=run_name + '_trial_' + str(i), nested=True) as child_run:
                
                # initialize prediction model
                decision_aware_reg = LinearRegression(input_dim=1,
                                                  output_dim=1)
                # intiialization
                decision_aware_reg.linear.weight.data = torch.tensor(reg.coef_).float()
                decision_aware_reg.linear.bias.data = torch.tensor(reg.intercept_).float()
                    
                if cfg.model.model_name == 'MSE':
                
                    loss_log_regret = pyepo.metric.regret(decision_aware_reg, opt_model, loader_test)
                    
                    mlflow.log_metrics({"bias": decision_aware_reg.linear.bias.data})
                    mlflow.log_metrics({"coef": decision_aware_reg.linear.weight.data})
                    mlflow.log_metrics({"trial_regret_last": loss_log_regret})
                    mlflow.log_metrics({"trial_regret_min": loss_log_regret})
                    
                    regret.append(loss_log_regret) 
                    min_regret.append(loss_log_regret)
                else:
                    # instantiate loss function
                    loss_func = hydra.utils.instantiate(cfg.model.model_func)
        
                    if cfg.model.model_name == 'DSL':
                        h = len(dataset_train)**cfg.model.h_exp
                        loss_func = loss_func(optmodel=opt_model,
                                              h=h,
                                              finite_diff_sch=cfg.model.finite_diff_sch,
                                              processes=cfg.model.processes)
                    else:
                        loss_func = loss_func(optmodel=opt_model)
                
                    logger.debug(f"loss_func: {loss_func}")

                    logger.info(f"decision aware reg coef start: {decision_aware_reg.linear.weight.data} and bias: {decision_aware_reg.linear.bias.data}")
                
                    if use_gpu==True:
                        logger.debug("Using GPU")
                        decision_aware_reg = decision_aware_reg.cuda()
                    else:
                        logger.debug("Using CPU")
                    # Check device of the first parameter
                    logger.debug(f"Model is on device: {next(decision_aware_reg.parameters()).device}")

                    # train model
                    loss_log, loss_log_regret = trainModel(reg=decision_aware_reg, 
                                                        loss_func=loss_func,
                                                        optmodel=opt_model,
                                                        loader_train=loader_train,
                                                        loader_test=loader_test,
                                                        use_gpu=use_gpu,
                                                        num_epochs=cfg.model.epochs)
                
                    mlflow.log_metrics({"bias": decision_aware_reg.linear.bias.data})
                    mlflow.log_metrics({"coef": decision_aware_reg.linear.weight.data})
                    mlflow.log_metrics({"trial_regret_last": loss_log_regret[-1]})
                    mlflow.log_metrics({"trial_regret_min": np.min(loss_log_regret)})
                    mlflow.log_metrics({"trial_loss": loss_log[-1]})
                    
                    regret.append(loss_log_regret[-1]) 
                    min_regret.append(np.min(loss_log_regret))
                    
                    logger.info(f"decision aware reg coef end: {decision_aware_reg.linear.weight.data} and bias: {decision_aware_reg.linear.bias.data}")
                
        mlflow.log_metrics({"avg_regret_last": np.mean(regret)})   
        mlflow.log_metrics({"avg_regret_min": np.mean(min_regret)})   
      
        
if __name__ == "__main__":
    main()