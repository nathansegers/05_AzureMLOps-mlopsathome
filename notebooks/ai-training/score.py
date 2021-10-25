import json
import numpy as np
import os
from tensorflow.keras.models import load_model

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'tveer-model')
    model = load_model(model_path)

def run(data):
    try:
        data = json.loads(data)
        conv_data = np.asarray(data['conv_data'])
        values = np.asarray(data['values'])
        result = model.predict([conv_data, values])
        return {"result": result.tolist()}
    except Exception as e:
        result = str(e)
        return {"error": result}