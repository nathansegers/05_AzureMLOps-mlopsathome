import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANIMALS = ['Cat', 'Dog', 'Panda'] # Animal names here

model = load_model('outputs/animal-cnn-test') # Model_name here!

@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    original_image = original_image.resize((64, 64))
    images_to_predict = np.expand_dims(np.array(original_image), axis=0)
    predictions = model.predict(images_to_predict)
    classifications = predictions.argmax(axis=1)

    return ANIMALS[classifications.tolist()[0]]