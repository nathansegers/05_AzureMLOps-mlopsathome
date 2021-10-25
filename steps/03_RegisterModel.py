import os
import sys
import json
import joblib
import argparse
import traceback
from dotenv import load_dotenv
from azureml.core import Run, Experiment, Workspace, Model
from azureml.core.authentication import AzureCliAuthentication
# For local development, set values in this section
load_dotenv()

def getConfiguration(details_file):
    try:
        with open(details_file) as f:
            config = json.load(f)
    except Exception as e:
        sys.exit(0)

    return config

def registerModel(model_name, description, run):

    model = run.register_model(model_name=model_name, model_path=f'outputs/{model_name}', tags={"runId": run.id}, description=description)
    print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, model.description, model.version))

    return model

def main():
    cli_auth = AzureCliAuthentication()

    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    model_name = os.environ.get("MODEL_NAME")
    model_description = os.environ.get("MODEL_DESCRIPTION")
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    config_state_folder = os.path.join(os.environ.get("ROOT_DIR"), 'config_states')

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    config = getConfiguration(config_state_folder + "/training-run.json")
    exp = Experiment(workspace=ws, name=experiment_name)
    run = Run(experiment=exp, run_id=config['runId'])
    model = registerModel(model_name, model_description, run)

    model_json = {}
    model_json["model"] = model.serialize()
    model_json["run"] = config

    print(model_json)

    with open(config_state_folder + "/model_details.json", "w") as model_details:
        json.dump(model_json, model_details)

if __name__ == '__main__':
    main()