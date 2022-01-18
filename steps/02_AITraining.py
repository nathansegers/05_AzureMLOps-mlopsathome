from ctypes import resize
from glob import glob
import json
import os
from datetime import datetime
import math
import random
import shutil
from typing import List, Tuple

from utils import connectWithAzure

from azureml.core import ScriptRunConfig, Experiment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

import cv2
from dotenv import load_dotenv



# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()

ANIMALS = os.environ.get('ANIMALS').split(',')
SEED = int(os.environ.get('RANDOM_SEED'))

INITIAL_LEARNING_RATE = float(os.environ.get('INITIAL_LEARNING_RATE')) # Float value
MAX_EPOCHS = int(os.environ.get('MAX_EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
PATIENCE = int(os.environ.get('PATIENCE'))
MODEL_NAME = os.environ.get('MODEL_NAME')

COMPUTE_NAME = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpu-cluster")
COMPUTE_MIN_NODES = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
COMPUTE_MAX_NODES = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
VM_SIZE = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

def prepareComputeCluster(ws):
    if COMPUTE_NAME in ws.compute_targets:
        compute_target = ws.compute_targets[COMPUTE_NAME]
        if compute_target and type(compute_target) is AmlCompute:
            print("found compute target: " + COMPUTE_NAME)
    else:
        print("creating new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = VM_SIZE,
                                                                    min_nodes = COMPUTE_MIN_NODES, 
                                                                    max_nodes = COMPUTE_MAX_NODES)

        # create the cluster
        compute_target = ComputeTarget.create(ws, COMPUTE_NAME, provisioning_config)
        
        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
        
        # For a more detailed view of current AmlCompute status, use get_status()
        print(compute_target.get_status().serialize())

        return compute_target

def prepareEnvironment(ws):

    # Create an Environment name for later use
    environment_name = os.environ.get('TRAINING_ENV_NAME')
    conda_dependencies_path = os.environ.get('CONDA_DEPENDENCIES_PATH')

    # It's called CondaDependencies, but you can also use pip packages ;-)
    # This is the old way to do so

    # env.python.conda_dependencies = CondaDependencies.create(
    #         # Using opencv-python-headless is interesting to skip the overhead of packages that we don't need in a headless-VM.
    #         pip_packages=['azureml-dataset-runtime[pandas,fuse]', 'azureml-defaults', 'tensorflow', 'scikit-learn', 'opencv-python-headless']
    #     )

    # We can directly create an environment from a saved file
    env = Environment.from_conda_specification(environment_name, file_path=conda_dependencies_path)
    env.python.user_managed_dependencies = not (os.environ.get('TRAIN_ON_LOCAL', 'False') == 'True') # True when training on local machine, otherwise False.
    # Register environment to re-use later
    env.register(workspace = ws)

    return env

def prepareTraining(ws, env, compute_target) -> Tuple[Experiment, ScriptRunConfig]:
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    script_folder = os.environ.get('SCRIPT_FOLDER')

    train_set_name = os.environ.get('TRAIN_SET_NAME')
    test_set_name = os.environ.get('TEST_SET_NAME')

    datasets = Dataset.get_all(workspace=ws) # Get all the datasets
    exp = Experiment(workspace=ws, name=experiment_name) # Create a new experiment

    args = [
        # You can set these to .as_mount() when not training on local machines, but this should also work.
    '--training-folder', datasets[train_set_name].as_download('./data/train'), # Currently, this will always take the last version. You can search a way to specify a version if you want to
    '--testing-folder', datasets[test_set_name].as_download('./data/test'), # Currently, this will always take the last version. You can search a way to specify a version if you want to
    '--max-epochs', MAX_EPOCHS,
    '--seed', SEED,
    '--initial-learning-rate', INITIAL_LEARNING_RATE,
    '--batch-size', BATCH_SIZE,
    '--patience', PATIENCE,
    '--model-name', MODEL_NAME]

    script_run_config = ScriptRunConfig(source_directory=script_folder,
                    script='train.py',
                    arguments=args,
                    compute_target=compute_target,
                    environment=env)


    print('Run started!')

    return exp, script_run_config

def downloadAndRegisterModel(ws, run):
    model_path = 'outputs/' + MODEL_NAME

    datasets = Dataset.get_all(workspace=ws) # Get all the datasets
    test_set_name = os.environ.get('TEST_SET_NAME')

    run.download_files(prefix=model_path)
    run.register_model(MODEL_NAME,
                model_path=model_path,
                tags={'animals': ','.join(ANIMALS), 'AI-Model': 'CNN', 'GIT_SHA': os.environ.get('GIT_SHA')},
                description="Image classification on animals",
                sample_input_dataset=datasets[test_set_name])

def main():
    ws = connectWithAzure()

    # We can also run on the local machine if we set the compute_target to None. We specify this in an ENV variable as TRAIN_ON_LOCAL.
    # If you don't give this parameter, we are defaulting to False, which means we will not train on local
    compute_target = None if os.environ.get('TRAIN_ON_LOCAL', 'False') == 'True' else prepareComputeCluster(ws)
    environment = prepareEnvironment(ws)
    exp, config = prepareTraining(ws, environment, compute_target)

    run = exp.submit(config=config)
    run.wait_for_completion(show_output=False) # We aren't going to show the training output, you can follow that on the Azure logs if you want to.
    print(f"Run {run.id} has finished.")

    downloadAndRegisterModel(ws, run)

if __name__ == '__main__':
    main()