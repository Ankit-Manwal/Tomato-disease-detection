from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn
import os

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can specify specific origins instead
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Load model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(parent_dir, 'saved_models/tomato.h5')
MODEL = tf.keras.models.load_model(model_path, compile=False)

# Class names
CLASS_NAMES = ['Tomato Bacterial spot',
                'Tomato Early blight',
                'Tomato Late blight',
                'Tomato Leaf Mold',
                'Tomato Septoria leaf spot',
                'Tomato Spider mites Two spottedspidermite',
                'Tomato Target Spot',
                'Tomato Tomato YellowLeaf Curl Virus',
                'Tomato Tomato mosaicvirus',
                'Tomato healthy']

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
    confidence = round(float(np.max(predictions[0]))*100, 2)
    return {
        'class': predicted_class,
        'confidence': confidence
     }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)