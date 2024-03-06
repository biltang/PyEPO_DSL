import os 
import logging 
from mlflow import MlflowClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        print(f"Created new experiment: {experiment_name} with ID {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} with ID {experiment_id}")
    
    return experiment_name