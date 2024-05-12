# Tomato-disease-detection

## Overview
This project aims to detect diseases in tomato plants using deep learning techniques. It includes a FastAPI backend for serving predictions and a simple frontend for uploading images and displaying the predicted disease and confidence.

## Backend
The backend is built using FastAPI, a modern, fast (high-performance), web framework for building APIs with Python 3.7+. It includes an endpoint `/predict` for receiving image uploads and returning predictions for the detected disease in tomatoes. The backend is trained with a deep learning model using TensorFlow/Keras.

### Setup
1. Install the necessary dependencies using `pip install -r requirements.txt`.
2. Run the backend server using `uvicorn main:app --reload`.

## Frontend
The frontend provides a simple user interface for uploading images and displaying prediction results. It is built using HTML, CSS, and JavaScript.

### Usage
1. Open `index.html` in a web browser.
2. Select an image of a tomato plant (in .jpg, .jpeg, .png, .gif, or .bmp format).
3. Click the "Upload" button to send the image to the backend for prediction.
4. View the predicted disease and confidence level displayed on the webpage.

## Model
The deep learning model used for disease detection is trained with a dataset containing images of healthy and diseased tomato plants. It is trained for 50 epochs and achieves an accuracy of approximately 93%.

### Training
The model is trained using TensorFlow and Keras, and the training process is not included in this repository. However, you can train your own model using the provided dataset and scripts.

## Troubleshooting
If you encounter any issues, such as "Failed to upload image"
Check File Selection: Ensure that you are selecting a valid image file (e.g., .jpg, .jpeg, .png, .gif) before clicking the upload button. Without a selected file, the upload process will fail.

Verify File Size: Make sure the size of the image file you are trying to upload is within the acceptable limits set by your backend server. If the file size exceeds the maximum allowed size, the upload process may fail.

Network Connection: Ensure that you have a stable internet connection. If your network connection is unstable or slow, it may cause issues with uploading files.

Cross-Origin Resource Sharing (CORS): Check if CORS is properly configured on your FastAPI backend to allow requests from the frontend domain or origin. If CORS is not configured correctly, the frontend may not be able to communicate with the backend.

Inspect Network Requests: Use the browser's developer tools to inspect the network requests when attempting to upload an image. Check if the request is being sent to the correct endpoint (http://localhost:8000/predict) and if any error responses are returned.

Check FormData Object: Double-check the creation of the FormData object in your frontend code to ensure that the file is being correctly appended to the FormData. You can log the FormData object just before sending the request to see if the file data is included.

Backend Debugging: Add logging or debugging statements to your FastAPI backend code to check if the image file is being received and processed correctly. This can help identify any issues with the backend code that might be causing the upload failure.

Try Different Image Files: If the issue persists with a specific image file, try uploading different image files to see if the problem is related to a particular file or file format.


Feel free to reach out if you have any questions or need further assistance!
