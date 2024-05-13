import logging 
from pathlib import Path 
import sys 
import os 
import shutil
from functools import partial
import time

import pickle
import hydra
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch
import mlflow
from mlflow import MlflowClient
import numpy as np
import pyepo 
from pyepo.metric.regret import calRegret
from hydra.core.hydra_config import HydraConfig

# add system path to src directory
sys.path.append('/home1/yongpeng/PyEPO_DSL/')
from utils.experiment_utils import setup_mlflow_experiment, create_path_if_not_exist, log_runtime

sys.path.append('/home1/yongpeng/PyEPO_DSL/simulation_test/shortest_path_reproduce/')
from pyepo_shortest_path_model import LinearRegression, trainModel

# TODO: why does adding this line allow us to find pkg.pyepo.func.dsl.DSLoss?
sys.path.append('/home1/yongpeng/PyEPO_DSL/simulation_test/')
from icml_mike_experiment.dsl_icml_data import wrap_pyepo_dataset # TODO: why does adding this line allow us to find pkg.pyepo.func.dsl.DSLoss?

# settings
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_dataset(file_path, optmodel, cfg):
    
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
    dataset_test_no_noise = pyepo.data.dataset.optDataset(optmodel, 
                                                dataset['test']['x'], 
                                                dataset['test']['c_no_noise'])
    
    # data loader        
    loader_train = DataLoader(dataset_train, 
                            batch_size=cfg.model.batch_size, 
                            shuffle=True)
    loader_val = DataLoader(dataset_val,
                            batch_size=cfg.model.batch_size,
                            shuffle=False)
    loader_test = DataLoader(dataset_test,
                            batch_size=cfg.model.batch_size,
                            shuffle=False)
    loader_test_no_noise = DataLoader(dataset_test_no_noise,
                            batch_size=cfg.model.batch_size,
                            shuffle=False)
    
    return loader_train, loader_val, loader_test, loader_test_no_noise
    
    
def construct_run_name(cfg, excluded_keys, mse_keys):
    
    # Initialize an empty list to collect the string components
    string_components = []
    mse_components = []
    
    # make sure early_stopping_params.enabled is the same as early_stopping used to record the parameter
    cfg.model.early_stopping = cfg.model.early_stopping_params.enabled
    
    for key, value in cfg.loss_func.items():
        # Check if the key is not in the list of excluded keys
        if key not in excluded_keys:
            # Construct the string component for this item
            component = f"{key}_{value}"
        
            # Append the component to the list
            string_components.append(component)

        if key in mse_keys:
            component = f"{key}_{value}"
            mse_components.append(component)

    for key, value in cfg.model.items():
        if key not in excluded_keys:
            # Construct the string component for this item
            component = f"{key}_{value}"
        
            # Append the component to the list
            string_components.append(component)
            
        if key in mse_keys:
            component = f"{key}_{value}"
            mse_components.append(component)
            
    # Join the components into a single string
    run_name = '-'.join(string_components)
    
    mse_name = '-'.join(mse_components)
    mse_name = mse_name.replace(cfg.loss_func.loss_name, 'MSE')
    
    return run_name, mse_name
    

def extract_date_time_hydra_dir(hydra_output_dir):
    
    date_part = os.path.basename(os.path.dirname(hydra_output_dir))  # Gets '2024-03-27'
    time_part = os.path.basename(hydra_output_dir)  # Gets '14-31-21'
    date_time = date_part + '/' + time_part
    hydra_output_base = hydra_output_dir.replace(date_time, '')
    
    return hydra_output_base, date_part, time_part


@log_runtime
def run_experiment_single_trial(i, cfg, run_name, model_params, use_gpu, path,  mse_name, track_experiment=False):
    """Single trial function so that we can run multiple trials in parallel if enabled

    Args:
        i (_type_): _description_
        cfg (_type_): _description_
        run_name (_type_): _description_
        model_params (_type_): _description_
        use_gpu (_type_): _description_
        path (_type_): _description_
        mse_name (_type_): _description_
        track_experiment (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # -------------------------------------------------------------------------------------
    # Experiment Set-up - For index i, get relevant data for trial i
    # -------------------------------------------------------------------------------------
    dataset_cfg = cfg.dataset
    
    optmodel = hydra.utils.instantiate(cfg.optmodel)
    
    # load test data 
    file = dataset_cfg.file + str(i) + dataset_cfg.file_extension
    file_path = path / file
    logger.info(f"data from {file_path} exists: {(file_path).exists()}")
    loader_train, loader_val, loader_test, loader_test_no_noise = load_dataset(file_path, optmodel, cfg)
    
    # set checkpoint path if early stopping is enabled
    if cfg.model.early_stopping_params.enabled:
        cfg.model.early_stopping_params.checkpoint_path = f"{cfg.model.early_stopping_params.checkpoint_path_orig}{dataset_cfg.dataset_path}/{run_name}/trial_{i}.pt"
    
    # get mse path if mse_init is enabled
    if cfg.model.mse_init:
        cfg.model.mse_init_params.mse_path = f"{cfg.model.mse_init_params.mse_path_orig}{dataset_cfg.dataset_path}/{mse_name}/trial_{i}.pt"
    
    # initialize prediction model
    # 5 input features, 40 output features - hard coded from pyepo paper
    # TODO: rewrite so that can use different prediction models based on config values
    decision_aware_reg = LinearRegression(input_dim=5, output_dim=40) 
    if use_gpu:
        decision_aware_reg = decision_aware_reg.cuda()
    
    # -------------------------------------------------------------------------------------
    # instantiate loss function
    # -------------------------------------------------------------------------------------
    loss_func = hydra.utils.instantiate(cfg.loss_func.loss_func)
    
    # loss function specific sethp
    h_schedule = False
    
    if cfg.loss_func.loss_name not in ['MSE', 'Cosine']:
        
        if cfg.model.mse_init:
            logging.info(f"Loading MSE check point at {cfg.model.mse_init_params.mse_path}")
        
            # Load the last checkpoint with the best model
            decision_aware_reg.load_state_dict(torch.load(cfg.model.mse_init_params.mse_path))
            
            init_val_regret = pyepo.metric.regret(decision_aware_reg, optmodel, loader_val)
            logging.info(f"Initial validation regret: {init_val_regret}")
        
        if cfg.loss_func.loss_name == 'DSL':
            h = len(loader_train.dataset)**cfg.loss_func.h_exp
            h_schedule = cfg.loss_func.h_sch
            loss_func = loss_func(optmodel=optmodel,
                                h=h,
                                finite_diff_sch=cfg.loss_func.finite_diff_sch,
                                processes=cfg.model.processes)
            
        elif cfg.loss_func.loss_name == 'SPO':
            loss_func = loss_func(optmodel=optmodel,
                                processes=cfg.model.processes)
            
            
        elif cfg.loss_func.loss_name in ['CosineSurrogate', 'CosineMSE']:
            loss_func = loss_func(**cfg.loss_func.parameters)
        
    logger.debug(f"loss_func: {loss_func}")
    # -------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------
    # Run Single Trial after loading data
    # -------------------------------------------------------------------------------------
    # train model
    start_time = time.time()
    loss_log, train_reg_log, val_reg_log = trainModel(reg=decision_aware_reg, 
                                        loss_func=loss_func,
                                        loss_name=cfg.loss_func.loss_name,
                                        optmodel=optmodel,
                                        loader_train=loader_train,
                                        loader_test=loader_val,
                                        use_gpu=use_gpu,
                                        num_epochs=cfg.model.epochs,
                                        lr=cfg.model.lr,
                                        h_schedule=h_schedule,
                                        lr_schedule=cfg.model.lr_schedule,
                                        early_stopping_cfg=cfg.model.early_stopping_params
                                        )
    elapsed_time = time.time() - start_time
    
    # reload the best model if early stopping is enabled
    if cfg.model.early_stopping_params.enabled:
        logging.info(f"Loading the best model from the last checkpoint at {cfg.model.early_stopping_params.checkpoint_path}")
        
        # Load the last checkpoint with the best model
        decision_aware_reg.load_state_dict(torch.load(cfg.model.early_stopping_params.checkpoint_path))
                
    # test metrics
    trial_train_loss = loss_log
    trial_train_regret = train_reg_log
    trial_val_regret = val_reg_log
    trial_test_regret = pyepo.metric.regret(decision_aware_reg, optmodel, loader_test)
    trial_test_regret_no_noise = pyepo.metric.regret(decision_aware_reg, optmodel, loader_test_no_noise)

    if track_experiment:
        with mlflow.start_run(run_name=run_name + '_trial_' + str(i), nested=True) as child_run:
            mlflow.log_metrics({"trial_train_loss": trial_train_loss})
            mlflow.log_metrics({"trial_train_regret": trial_train_regret})
            mlflow.log_metrics({"trial_val_regret": trial_val_regret})
            mlflow.log_metrics({"trial_test_regret": trial_test_regret})
            mlflow.log_metrics({"trial_test_regret_no_noise": trial_test_regret_no_noise})
            mlflow.log_metrics({"trial_execution_time": elapsed_time})
    
    return trial_train_loss, trial_train_regret, trial_val_regret, trial_test_regret, trial_test_regret_no_noise


def run_experiment_many_trials(cfg, run_name, model_params, use_gpu, path, mse_name, track_experiment=False):
    """Separate function to run multiple trials of the experiment for handling multi-core processing case

    Args:
        cfg (_type_): _description_
        run_name (_type_): _description_
        model_params (_type_): _description_
        use_gpu (_type_): _description_
        path (_type_): _description_
        mse_name (_type_): _description_
        track_experiment (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    dataset_cfg = cfg.dataset
    
    loss = []
    train_regret = []
    val_regret = []
    regret = []
    regret_no_noise = []
            
    if cfg.general.multi_core_cpu:
        logger.info("multi-core")
        logging.info(f"Number of cores: {mp.cpu_count()}")
        partial_run_experiment = partial(run_experiment_single_trial, 
                                       cfg=cfg,
                                       run_name=run_name, 
                                       model_params=model_params, 
                                       use_gpu=use_gpu, 
                                       path=path,  
                                       mse_name=mse_name,
                                       track_experiment=track_experiment)
            
        
        test_indices = range(0, dataset_cfg.num_trials)
        with ProcessPoolExecutor(max_workers=cfg.general.max_workers) as executor:
            results = list(executor.map(partial_run_experiment, test_indices))
                
            for i in test_indices:
                trial_train_loss, trial_train_regret, trial_val_regret, trial_test_regret, trial_test_regret_no_noise = results[i]
                logging.info(f"Trial {i} train loss: {trial_train_loss} train regret: {trial_train_regret} val regret: {trial_val_regret} test regret: {trial_test_regret} test regret no noise: {trial_test_regret_no_noise}")
                    
                loss.append(trial_train_loss) 
                train_regret.append(trial_train_regret)
                val_regret.append(trial_val_regret) 
                regret.append(trial_test_regret) 
                regret_no_noise.append(trial_test_regret_no_noise)
                    
    else:
        logger.info("single-core")
        for i in range(dataset_cfg.num_trials):
            trial_train_loss, trial_train_regret, trial_val_regret, trial_test_regret, trial_test_regret_no_noise = run_experiment_single_trial(i,
                                                                                                            cfg, 
                                                                                                            run_name, 
                                                                                                            model_params, 
                                                                                                            use_gpu, 
                                                                                                            path,  
                                                                                                            mse_name,
                                                                                                            track_experiment=track_experiment)
                
            logging.info(f"Trial {i} train loss: {trial_train_loss} train regret: {trial_train_regret} val regret: {trial_val_regret} test regret: {trial_test_regret} test regret no noise: {trial_test_regret_no_noise}")
            
            loss.append(trial_train_loss)
            train_regret.append(trial_train_regret) 
            val_regret.append(trial_val_regret) 
            regret.append(trial_test_regret) 
            regret_no_noise.append(trial_test_regret_no_noise)
            
    return loss, train_regret, val_regret, regret, regret_no_noise

            
@hydra.main(version_base=None, config_path="configs/pyepo_shortest_path_reproduce/", config_name="shortest_path_reproduce_week_4_1_exp")
#@hydra.main(version_base=None, config_path="configs/icml/", config_name="icml_reproduce")
def main(cfg: DictConfig) -> None:
     
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    hydra_base_dir, date, time = extract_date_time_hydra_dir(hydra_output_dir=hydra_output_dir)
    
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
    
    # -------------------------------------------------------------------------------------
    # Set up experiment name and run name 
    # -------------------------------------------------------------------------------------
    cur_data_instance_name = dataset_cfg.dataset_path
    experiment_name = f"{dataset_cfg.setup}_{cur_data_instance_name}" 
    
    # set run name
    excluded_keys = ['loss_func', 'mse_init_params', 'early_stopping_params'] # exclude these keys in constructing the run name
    mse_keys = ['loss_name', 'mse_init', 'batch_size', 'lr', 'lr_schedule', 'processes', 'epochs', 'early_stopping']
    
    run_name, mse_name = construct_run_name(cfg=cfg, excluded_keys=excluded_keys, mse_keys=mse_keys)
    logging.info(f"Experiment name: {experiment_name} and run name: {run_name}")
     
    # set run parameters
    model_params = {**cfg.model, **cfg.loss_func}
    for e in excluded_keys:
        not_found = model_params.pop(e, 'Not Found') 
    
    # -------------------------------------------------------------------------------------    
    
    # -------------------------------------------------------------------------------------
    # Run Experiment
    # -------------------------------------------------------------------------------------
    
    if cfg.general.track_experiment: # If experiment tracking is enabled - MLflow setup 
        logger.info("Experiment tracking is enabled")

        # set mlflow experiment
        if cfg.mlflow.local == False:
            # instantiate credentials    
            os.environ['MLFLOW_TRACKING_USERNAME'] = cfg.cred.username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = cfg.cred.password
    
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        experiment_name = setup_mlflow_experiment(client=MlflowClient(tracking_uri=cfg.mlflow.tracking_uri),
                                                root_path=cfg.mlflow.root_path,
                                                experiment_name=experiment_name)
        mlflow.set_experiment(experiment_name)    
        # -------------------------------------------------------------------------------------
        
        with mlflow.start_run(run_name=run_name) as run:
            # log params 
            mlflow.log_params(model_params)
        
            # set gpu tag
            mlflow.set_tag("gpu", use_gpu)
            
            loss, train_regret, val_regret, regret, regret_no_noise = run_experiment_many_trials(cfg, run_name, model_params, use_gpu, path, mse_name, track_experiment=cfg.general.track_experiment)
            mlflow.log_metrics({"avg_train_loss": np.mean(loss)})
            mlflow.log_metrics({"avg_train_regret": np.mean(train_regret)})
            mlflow.log_metrics({"avg_val_regret": np.mean(val_regret)})
            mlflow.log_metrics({"avg_test_regret": np.mean(regret)}) 
            mlflow.log_metrics({"avg_test_regret_no_noise": np.mean(regret_no_noise)})
            
    else: # If no experiment tracking, just run code
        loss, train_regret, val_regret, regret, regret_no_noise = run_experiment_many_trials(cfg, run_name, model_params, use_gpu, path, mse_name, track_experiment=cfg.general.track_experiment)
        
    # -------------------------------------------------------------------------------------
            
    # -------------------------------------------------------------------------------------
    # Move hydra output folder
    # -------------------------------------------------------------------------------------
    new_hydra_log_path = f"{hydra_base_dir}{experiment_name}/{run_name}/{date}"
    create_path_if_not_exist(new_hydra_log_path)
    
    logger.debug(f"moving hydra output folder from {hydra_output_dir} to {new_hydra_log_path}")
    
    # Move the directory
    shutil.move(hydra_output_dir, new_hydra_log_path)
    
    # move slurm output file
    if cfg.general.slurm_job_name is not None:
        slurm_output_file = f"{cfg.general.slurm_output}{cfg.general.slurm_job_name}_{cfg.general.slurm_job_id}.out"
        slurm_err_file = f"{cfg.general.slurm_output}{cfg.general.slurm_job_name}_{cfg.general.slurm_job_id}.err"
    
        # Move the directory
        new_hydra_log_path = f"{new_hydra_log_path}/{time}"
        logger.debug(f"moving slurm output file from {slurm_output_file} to {new_hydra_log_path}")
    
        shutil.move(slurm_output_file, new_hydra_log_path)
        shutil.move(slurm_err_file, new_hydra_log_path)
              
            
if __name__ == "__main__":
    main()