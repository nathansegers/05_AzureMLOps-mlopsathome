import os
import sys
import json
import uuid
import joblib
import argparse
import traceback
from dotenv import load_dotenv

from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Run, Experiment, Workspace, Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import Webservice, AciWebservice
from azureml.core.authentication import AzureCliAuthentication

load_dotenv()

def getConfiguration(details_file):
    try:
        with open(details_file) as f:
            config = json.load(f)
    except Exception as e:
        print(e)
        sys.exit(0)

    return config

def main():
    cli_auth = AzureCliAuthentication()
    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    environment = os.environ.get("AML_ENV_NAME")

    temp_state_directory = os.environ.get("TEMP_STATE_DIRECTORY")
    score_script_path = os.path.join(os.environ.get("ROOT_DIR"), 'scripts', 'score.py')

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    config = getConfiguration(temp_state_directory + "/model_details.json")

    model = Model.deserialize(workspace=ws, model_payload=config['model'])
    
    env = Environment(environment + '-deployment')
    cd = CondaDependencies.create(
        pip_packages=['azureml-defaults','numpy', 'tensorflow']
    )

    env.python.conda_dependencies = cd
    env.register(workspace = ws)
    inference_config = InferenceConfig(entry_script=score_script_path, environment=env)

    # Azure Container Instance 
    aciconfig = AciWebservice.deploy_configuration(
        cpu_cores=1, 
        memory_gb=1, 
        tags={"data": "mnist",  "method" : "keras"}, 
        description='Classify handwritten digits'
    )

    service_name = 'mnist-digits-svc-' + str(uuid.uuid4())[:4]
    service = Model.deploy(workspace=ws, 
                        name=service_name, 
                        models=[model], 
                        inference_config=inference_config, 
                        deployment_config=aciconfig)

    service.wait_for_deployment(show_output=True)

    with open(temp_state_directory + "/service_details.json", "w") as service_details:
        json.dump(service.serialize(), service_details)
    

    
    


if __name__ == '__main__':
    main()