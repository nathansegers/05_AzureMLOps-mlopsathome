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


# For local development, set values in this section
load_dotenv()

def main():
    cli_auth = AzureCliAuthentication()

    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    input_dataset_name = os.environ.get("INPUT_DATASET_NAME")
    training_testing_dataset_name = os.environ.get("TRAINING_TESTING_DATASET")
    generator_dataset_name = os.environ.get("GENERATOR_DATASET_NAME")

    script_folder = os.path.join(os.environ.get('ROOT_DIR'), 'scripts')
    config_state_folder = os.path.join(os.environ.get('ROOT_DIR'), 'config_states')

    train_test_data_folder = os.path.join(os.environ.get('ROOT_DIR'), 'data/tmp/train_test_data')
    os.makedirs(train_test_data_folder, exist_ok=True)

    generator_folder = os.path.join(os.environ.get('ROOT_DIR'), 'data/tmp/generator')
    os.makedirs(generator_folder, exist_ok=True)

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    datastore = Datastore(ws)

    # Prepare!
    all_labels_np, all_components_np, all_sizes_np, components = loadInputData(input_dataset_name, ws)

    X_train_conv, X_train_values, X_test_conv, X_test_values, y_train, y_test = trainTestSplit(all_labels_np, all_components_np, all_sizes_np)

    saveNumpyArrays(train_test_data_folder, X_train_conv=X_train_conv, X_train_values=X_train_values, X_test_conv=X_test_conv, X_test_values=X_test_values, y_train=y_train, y_test=y_test)

    generateData(datastore, ws, generator_dataset_name, generator_folder, components, X_train_conv, X_train_values, y_train)

    y_train_generated, X_train_values_generated, X_train_conv_generated = processGeneratedData(generator_folder)

    saveNumpyArrays(train_test_data_folder, y_train_generated=y_train_generated, X_train_values_generated=X_train_values_generated, X_train_conv_generated=X_train_conv_generated)
    saveNumpyArrays(train_test_data_folder, component_names=np.asarray(components))

    datastore.upload(src_dir=train_test_data_folder, target_path='train_test_data')
    train_test_data = Dataset.File.from_files(
        [
            (datastore, 'train_test_data')
        ],
        validate=False
    )
    train_test_data.register(
        workspace=ws,
        name=training_testing_dataset_name,
        description="A part of the components that have been processed and split in training and testing sets for an AI model.",
        create_new_version=True
    )
    

def processGeneratedData(generator_folder):
    all_generated_images = []
    all_generated_sizes = []
    all_generated_labels = []
    for img in sorted(glob(f"{generator_folder}/*.png")):
        size_name = img.replace('.png', '--size.json')
        all_generated_images.append(plt.imread(img)[:,:,:3] / 255)
        all_generated_labels.append(img.split('-')[-3].split('/')[-1]) # Get the Object name (data/tveer/generator/Object 1-0-1.png --> Object 1)
        with open(size_name, 'r') as f:
            all_generated_sizes.append(json.load(f))

    X_train_conv_generated = np.asarray(all_generated_images)
    X_train_values_generated = np.asarray(all_generated_sizes)
    all_generated_labels = np.asarray(all_generated_labels)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(all_generated_labels)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_train_generated = onehot_encoder.fit_transform(integer_encoded)
    return y_train_generated, X_train_values_generated, X_train_conv_generated

def generateData(datastore, ws, generator_dataset_name, generator_folder, components, X_train_conv, X_train_values, y_train):
    train_generator = ImageDataGenerator(rotation_range=360)
    time_to_repeat_generator = 5
    image_generator = train_generator.flow(
        [
            np.repeat(X_train_conv, time_to_repeat_generator, 0),
            np.repeat(X_train_values, time_to_repeat_generator, 0)
        ],
        np.repeat(y_train, time_to_repeat_generator, 0),
        batch_size = 32
    )

    for i in range(20):
        generated_data, generated_labels = image_generator.next()
        for j in range(generated_data[1].shape[0]):
            plt.imsave(f"{generator_folder}/{components[np.argmax(generated_labels[j])]}-{i}-{j}.png", generated_data[0][j])
            with open(f"{generator_folder}/{components[np.argmax(generated_labels[j])]}-{i}-{j}--size.json", "w") as f:
                json.dump(generated_data[1][j], f)

    datastore.upload(src_dir=generator_folder, target_path='generator_data')
    generator_data = Dataset.File.from_files(
        [
            (datastore, 'generator_data')
        ],
        validate=False
    )
    generator_data.register(
        workspace=ws,
        name=generator_dataset_name,
        description="Components of the 't Veer dataset which have been generated and slightly augmented with rotations.",
        create_new_version=True
    )

def saveNumpyArrays(folder, **arrays):
    for array_name, array in arrays.items():
        np.save(f"{folder}/{array_name}.npy", array)


def trainTestSplit(all_labels_np, all_components_np, all_sizes_np):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(sorted(all_labels_np))

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    training_indices = []
    test_indices = []
    for obj in range(0, onehot_encoded.shape[1]):
        obj_indices = np.where(onehot_encoded[:,obj] == 1)[0]
        np.random.shuffle(obj_indices)
        training_samples = math.floor(0.7 * len(obj_indices)) ## Take 70 % training samples.
        training_indices.extend(obj_indices[:training_samples])
        test_indices.extend(obj_indices[training_samples:])

    # print(f"{len(training_indices)} training indices")
    # print(f"{len(test_indices)} test indices")

    X_train_conv = all_components_np[training_indices]
    X_train_values = all_sizes_np[training_indices]
    X_test_conv = all_components_np[test_indices]
    X_test_values = all_sizes_np[test_indices]

    y_train = onehot_encoded[training_indices]
    y_test = onehot_encoded[test_indices]
    return X_train_conv, X_train_values, X_test_conv, X_test_values, y_train, y_test


def loadInputData(input_dataset_name, ws):
    input_dataset = Dataset.get_by_name(workspace=ws, name=input_dataset_name)
    
    temp_directory = os.path.join(os.environ.get('ROOT_DIR'), f'data/tmp/{input_dataset_name}')
    os.makedirs(temp_directory, exist_ok=True)
    
    moount_context = input_dataset.mount(temp_directory)
    moount_context.start()

    components = os.listdir(temp_directory)

    all_components = []
    all_labels = []
    all_sizes = []
    for comp in components:
        for img_uri in glob(os.path.join(temp_directory, comp) + "/*.png"):
            try:
                size = img_uri.split(".png")[-2]
                with open(size + '--size.json', "r") as f:
                    all_sizes.append(json.load(f))
                    img = plt.imread(img_uri)[:,:,:3]
                    all_components.append(img)
                    all_labels.append(comp)
            except FileNotFoundError:
                pass

    all_labels_np = np.array(all_labels)
    all_components_np = np.array(all_components)
    all_sizes_np = np.array(all_sizes)

    return all_labels_np, all_components_np, all_sizes_np, components

if __name__ == '__main__':
    main()
