import os
import sys
import json
import joblib
import argparse
import traceback
from dotenv import load_dotenv
from azureml.core import ScriptRunConfig
from azureml.core.environment import Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Run, Experiment, Workspace, Dataset, Datastore

# For local development, set values in this section
load_dotenv()

def prepareEnv(ws, env_name):
    env = Environment(env_name)
    cd = CondaDependencies.create(
        pip_packages=['azureml-dataset-runtime[pandas,fuse]', 'azureml-defaults', 'scikit-learn', 'tensorflow'],
        )

    env.python.conda_dependencies = cd

    # Register environment to re-use later
    env.register(workspace = ws)

    return env

def prepareMachines(ws):
    ## If machine not yet ready, create !
    # choose a name for your cluster
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    compute_min_nodes = int(os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES"))
    compute_max_nodes = int(os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES"))
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU")

    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print("Found compute target, will use this one: " + compute_name)
    else:
        print("creating new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size, min_nodes = compute_min_nodes, max_nodes = compute_max_nodes)
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    return compute_target

def prepareTraining(dataset, model_name, script_folder, compute_target, env):
    args = ['--data-folder', dataset.as_mount(), '--epochs', 30, '--batch_size', 32, '--model_name', model_name]
    src = ScriptRunConfig(source_directory=script_folder, script='train.py', arguments=args, compute_target=compute_target, environment=env)

    return src

def main():
    cli_auth = AzureCliAuthentication()

    workspace_name = os.environ.get("WORKSPACE_NAME")
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    env_name = os.environ.get("AML_ENV_NAME")
    model_name = os.environ.get("MODEL_NAME")

    dataset_name = os.environ.get("TRAINING_TESTING_DATASET")

    script_folder = os.path.join(os.environ.get('ROOT_DIR'), 'scripts')
    config_state_folder = os.path.join(os.environ.get('ROOT_DIR'), 'config_states')

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )
    print(dataset_name)

    # Prepare!
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
    
    compute_target = prepareMachines(ws)
    env = prepareEnv(ws, env_name)
    src = prepareTraining(dataset, model_name, script_folder, compute_target, env)

    ## Start training
    exp = Experiment(workspace=ws, name=experiment_name)
    run = exp.submit(config=src)

    run.wait_for_completion()

    run_details = {k:v for k,v in run.get_details().items() if k not in ['inputDatasets', 'outputDatasets']}
    
    with open(config_state_folder + '/training-run.json', 'w') as training_run_json:
        json.dump(run_details, training_run_json)
    

if __name__ == '__main__':
    main()
