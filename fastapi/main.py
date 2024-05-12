# '''from fastapi import FastAPI
# import uvicorn                         

# app = FastAPI()

# @app.get("/ping")
# async def ping():
#     return "working....properly"

# if __name__ == "__main__":                                    # to run it directly without cmd  "uvicorn main:app --reload"
#     uvicorn.run(app, host='localhost', port=8000)

# '''
# import pickle

# from fastapi import FastAPI, File, UploadFile, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image          # to read imagesP
# import tensorflow as tf
 
# app = FastAPI()

# ##########################################################################################################
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# #############################################################################################################
# #model upload
# PATH=r'D:\college\xxxxxxxxxxx\tomato-diease-detection\saved_models\model1.h5'
# MODEL = tf.keras.models.load_model(PATH,compile=False)

# CLASS_NAMES = ['Tomato_Bacterial_spot',
#                 'Tomato_Early_blight',
#                 'Tomato_Late_blight',
#                 'Tomato_Leaf_Mold',
#                 'Tomato_Septoria_leaf_spot',
#                 'Tomato_Spider_mites_Two_spotted_spider_mite',
#                 'Tomato__Target_Spot',
#                 'Tomato__Tomato_YellowLeaf__Curl_Virus',
#                 'Tomato__Tomato_mosaic_virus',
#                 'Tomato_healthy']

# ##############################################################################################################################
# #read file fn
# def read_file_as_image(data) -> np.ndarray:           # data=bytes          #  BytesIO(data) =to supply bytes to pillow
#     image = np.array(Image.open(BytesIO(data)))       # Image.open()= convert bytes into pillow image     
#     return image                                      # np.array()= image to array

# ##############################################################################################################################
# #predict function input
# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)    # user have to sent file/image as a parameter
# ):
#     image = read_file_as_image(await file.read())  # "await" for many user at once

#     img_batch = np.expand_dims(image, 0)     #add 1 more dimension [[256,256,3]] as model takes up batch of images only
    
#     predictions = MODEL.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': str(predicted_class),
#         'confidence': float(confidence)
#     }

# #########################################################################################

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)


from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn

app = FastAPI()


@app.get("/ping")
async def ping():
    return "working....properly"




# Load model
PATH = 'D:/college/xxxxxxxxxxx/tomato-diease-detection/saved_models/model1.h5'
MODEL = tf.keras.models.load_model(PATH, compile=False)

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

