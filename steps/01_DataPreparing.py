import argparse
import json
import os
import sys
import traceback
from glob import glob
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
from azureml.core import Dataset, Datastore, Experiment, Run, Workspace
from azureml.core.authentication import AzureCliAuthentication
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
  
from azureml.core import Dataset
from azureml.opendatasets import MNIST

# For local development, set values in this section
load_dotenv()

def downloadData(data_folder, ws):

    dataset_name = os.environ.get("DATASET_NAME")
    dataset_description = os.environ.get("DATASET_DESCRIPTION")
    dataset_new_version = os.environ.get("DATASET_NEW_VERSION") == 'true'


    mnist_file_dataset = MNIST.get_file_dataset()
    mnist_file_dataset.download(data_folder, overwrite=True)

    mnist_file_dataset = mnist_file_dataset.register(workspace=ws,
                                                    name=dataset_name,
                                                    description=dataset_description,
                                                    create_new_version=dataset_new_version)

    return {
        'name': dataset_name,
        'description': dataset_description,
    }

def main():
    print("Everything is working")
    cli_auth = AzureCliAuthentication()

    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    temp_state_directory = os.environ.get("TEMP_STATE_DIRECTORY")

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    data_folder = os.path.join(os.getcwd(), os.environ.get("DATA_FOLDER"))
    os.makedirs(data_folder, exist_ok=True)

    mnist_file_dataset = downloadData(data_folder, ws)

    # return_object = {
    #     **mnist_file_dataset
    # }
    os.makedirs(temp_state_directory, exist_ok=True)

    with open(os.path.join(temp_state_directory, 'dataset.json'), 'w') as dataset_json:
        json.dump(mnist_file_dataset, dataset_json)

if __name__ == '__main__':
    main()
