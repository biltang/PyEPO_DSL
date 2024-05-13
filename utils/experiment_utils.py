import os 
import logging
from collections import defaultdict
import time
from functools import wraps
from datetime import datetime

import pandas as pd
from pandas import DataFrame
from mlflow import MlflowClient
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def log_runtime(func):
    """Decorator to log the runtime of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time} seconds to run.")
        return result
    return wrapper


def create_path_if_not_exist(path: str) -> None:
    
    # Check if the directory exists
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)
        
        logger.info(f"Directory '{path}' has been created.")
        
    else:
        logger.info(f"Directory '{path}' already exists.")


def setup_mlflow_experiment(client: MlflowClient, root_path: str, experiment_name: str, experiment_tags: dict=None) -> str:
    """Create a new mlflow experiment if it does not exist"""
    experiment_name = f"{root_path}{experiment_name}"
    
    # Check if the experiment already exists
    experiment = client.get_experiment_by_name(name=experiment_name)
    
    if experiment is None:
        # Create the experiment if it doesn't exist
        experiment_id = client.create_experiment(name=experiment_name,
                                               tags=experiment_tags)
        logger.info(f"Created new experiment: {experiment_name} with ID {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} with ID {experiment_id}")
    
    return experiment_name


@log_runtime
def pull_mlflow_local_experiments(experiment_folder: str, experiment_name: str, keep_runs_with_k_trials: int=None, parent_runs_keep_columns: list=None, trial_runs_keep_columns: list=None):
    """Pull all runs from a local mlflow experiment and return a dataframe with parent and child runs merged together.

    Args:
        experiment_folder (str): local experiment folder path
        experiment_name (str): name of the experiment
        keep_runs_with_k_trials (int, optional): Whether to filter runs based on number of completed child/trial runs. Defaults to None.
        parent_runs_keep_columns (list, optional): Which columns to keep from parent runs. Defaults to None.
        trial_runs_keep_columns (list, optional): Which columns to keep from child runs. Defaults to None.

    Returns:
        pandas.DataFrame: dataframe with parent and child runs merged together
    """
    mlflow.set_tracking_uri(experiment_folder)
    
    # search for the experiment name
    experiments = mlflow.search_experiments(filter_string=f"name='{experiment_name}'")
    experiment_ids = [e.experiment_id for e in experiments] # get experiment ids with experiment_name so we can query runs
    
    # get all runs for the experiment
    
    df = mlflow.search_runs(experiment_ids)
    df = df[df.status=='FINISHED'].sort_values(by='end_time', ascending=True) # only get finished runs
    
    # Set experiment name, degree, and half-width
    df['experiment_name'] = experiment_name
    df['degree'] = df['experiment_name'].str.extract(r'deg_(\d+)').astype('int')  # get degree
    df['half_width'] = df['experiment_name'].str.extract(r'e_([0-9]+\.[0-9]+)').astype('float') # get half-width e
    
    #------------------------------------------------------------------------
    # parent runs
    #------------------------------------------------------------------------
    
    if keep_runs_with_k_trials is not None:
        parent_runs_ct = df.groupby('tags.mlflow.parentRunId').size()
        keep_ids = parent_runs_ct[parent_runs_ct >= keep_runs_with_k_trials].index.values # get parent run ids with at least k trials
        
        parent_runs = df[df.run_id.isin(keep_ids)] # keep only runs with at least k trials based on parent run id
        parent_runs = parent_runs.drop_duplicates(subset=['tags.mlflow.runName'], keep='last') # keep only the last run for each parent run name in case of duplicates
        
    # parent cols keep
    if parent_runs_keep_columns is None:
        param_cols = [c for c in df.columns if 'params' in c]
        metric_cols = ['metrics.avg_train_loss','metrics.avg_train_regret', 'metrics.avg_val_regret','metrics.avg_test_regret', 'metrics.avg_test_regret_no_noise']
        parent_cols = ['run_id', 'start_time', 'experiment_name', 'degree', 'half_width', 'tags.mlflow.runName'] + param_cols + metric_cols
        
    parent_runs = parent_runs[parent_cols]
    
    #------------------------------------------------------------------------
    # child runs
    #------------------------------------------------------------------------
    
    # child trial runs columns to keep
    if trial_runs_keep_columns is None:
        child_run_keep_cols = ['tags.mlflow.runName', 'tags.mlflow.parentRunId', 'experiment_name', 
                               'metrics.trial_train_loss', 'metrics.trial_train_regret', 'metrics.trial_val_regret', 'metrics.trial_test_regret', 'metrics.trial_test_regret_no_noise']
    
    # get child runs
    parent_run_ids = parent_runs['run_id'].values
    child_runs = df[df['tags.mlflow.parentRunId'].isin(parent_run_ids)] # keep only child runs corresponding to parent runs with at least k trials
    child_runs = child_runs[child_run_keep_cols].rename(columns={'tags.mlflow.runName': 'child_run_name'})
    
    # merge parent and child runs    
    merged_runs = parent_runs.merge(child_runs.drop(columns=['experiment_name']), left_on='run_id', right_on='tags.mlflow.parentRunId')
    
    return merged_runs
        

def aggregate_mlflow_local_experiments(experiment_folder: str, keep_runs_with_k_trials: int=None, parent_runs_keep_columns: list=None, trial_runs_keep_columns: list=None):
    """ For all experiments in a local mlflow experiment folder, pull all runs and merge parent and child runs together into a single dataframe.

    Args:
        experiment_folder (str): local experiment folder path
        keep_runs_with_k_trials (int, optional): Whether to filter runs based on number of completed child/trial runs. Defaults to None.
        parent_runs_keep_columns (list, optional): Which columns to keep from parent runs. Defaults to None.
        trial_runs_keep_columns (list, optional): Which columns to keep from child runs. Defaults to None.

    Returns:
        pandas.DataFrame: dataframe with all runs from all experiments merged together from a local mlflow experiment folder
    """
    mlflow.set_tracking_uri(experiment_folder)
    
    experiments = mlflow.search_experiments() # get all experiments in local experiment folder
    
    # get all experiment names to search through - get count in case duplicates created so we can track this
    exp_names = defaultdict(int)
    exp_ids = defaultdict(list)
    for experiment in experiments:
        exp_names[experiment.name] += 1
        exp_ids[experiment.name].append(experiment.experiment_id)
    
    all_runs_pd = None
    for experiment_name in exp_names.keys():
    
        logger.info(f"Name: {experiment_name}")
        cur_pd = pull_mlflow_local_experiments(experiment_folder=experiment_folder, 
                                            experiment_name=experiment_name,
                                            keep_runs_with_k_trials=100,
                                            parent_runs_keep_columns=parent_runs_keep_columns,
                                            trial_runs_keep_columns=trial_runs_keep_columns)
    
        if all_runs_pd is None:
            all_runs_pd = cur_pd
        else:
            all_runs_pd = pd.concat([all_runs_pd, cur_pd], ignore_index=True)
            
    all_runs_pd['trial_num'] = all_runs_pd['child_run_name'].str.split('_').str[-1].astype(int)

    return all_runs_pd, exp_ids


def save_clean_aggregated_mlflow_local_experiments_to_csv(save_folder: str, save_name: str, experiment_folder: str, keep_runs_with_k_trials: int=100, parent_runs_keep_columns: list=None, trial_runs_keep_columns: list=None):
    """ Save aggregated runs from all experiments in a local mlflow experiment folder to a csv file.

    Args:
        save_folder (str): folder to save the csv file
        save_name (str): name of the csv file
        experiment_folder (str): local experiment folder path
        keep_runs_with_k_trials (int, optional): Whether to filter runs based on number of completed child/trial runs. Defaults to 100.
        parent_runs_keep_columns (list, optional): Which columns to keep from parent runs. Defaults to None.
        trial_runs_keep_columns (list, optional): Which columns to keep from child runs. Defaults to None.

    Returns:
        pandas.DataFrame: dataframe with all runs from all experiments merged together from a local mlflow experiment folder
    """
    all_runs_pd, exp_ids = aggregate_mlflow_local_experiments(experiment_folder=experiment_folder, 
                                                     keep_runs_with_k_trials=keep_runs_with_k_trials,
                                                     parent_runs_keep_columns=parent_runs_keep_columns,
                                                     trial_runs_keep_columns=trial_runs_keep_columns)
    
    # save aggregated runs to csv
    # Get today's date
    today = datetime.today()
    date_string = today.strftime('%Y_%m_%d') # Convert to string in the format YYYY-MM-DD

    all_runs_pd.to_csv(f"{save_folder}/{save_name}_{date_string}_all_runs.csv", index=False)
    logger.info(f"Saved aggregated runs to {save_folder}/{save_name}_{date_string}_all_runs.csv")
    
    # delete runs by experiment id
    for exp_id in exp_ids.values():
        clean_mlflow_local_experiments(experiment_folder=experiment_folder, experiment_ids=exp_id)
        
    return all_runs_pd
    
    
def clean_mlflow_local_experiments(experiment_folder: str, experiment_name: str =None, experiment_ids: list=None):
    """ Delete all runs for a given experiment name or experiment ids.

    Args:
        experiment_folder (str): local experiment folder path
        experiment_name (str, optional): Name of experiment. Defaults to None.
        experiment_ids (list, optional): Experiment Ids of Experiment. Defaults to None.

    Raises:
        ValueError: Either experiment_ids or experiment_name must be provided.
    """
    mlflow.set_tracking_uri(experiment_folder)
    
    if experiment_ids is None and experiment_name is None:
        raise ValueError("Either experiment_ids or experiment_name must be provided.")
    elif experiment_ids is not None:
        runs = mlflow.search_runs(experiment_ids=experiment_ids)
    else:
        experiments = mlflow.search_experiments(filter_string=f"name='{experiment_name}'")
        experiment_ids = [e.experiment_id for e in experiments]
        runs = mlflow.search_runs(experiment_ids)
        
    # Delete all the runs
    for run in runs.iterrows():
        run_id = run[1]["run_id"]
        mlflow.delete_run(run_id)
    
    logger.info(f"Deleted all runs for experiment: {experiment_name} with IDs {experiment_ids}")
    
    
# TODO: Implement this function    
def add_new_experiment_runs_to_aggregate():
    """Add new runs to the aggregate"""
    pass

