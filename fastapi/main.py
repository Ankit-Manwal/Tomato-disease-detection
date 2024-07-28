from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn
import os

app = FastAPI()


@app.get("/ping")
async def ping():
    return "working....properly"




# Load model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

model_path = os.path.join(parent_dir, 'saved_models/tomato.h5')
MODEL = tf.keras.models.load_model(model_path, compile=False)

# Class names
CLASS_NAMES = ['Tomato_Bacterial_spot',
                'Tomato_Early_blight',
                'Tomato_Late_blight',
                'Tomato_Leaf_Mold',
                'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot',
                'Tomato__Tomato_YellowLeaf__Curl_Virus',
                'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy']

# Function to read image file
def read_file_as_image(data):
    image = np.array(Image.open(data.file))
    return image

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(file)
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {
        'class': predicted_class,
        'confidence': confidence
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

