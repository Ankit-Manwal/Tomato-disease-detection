
import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

def main():
    # Set page configuration
    st.set_page_config(page_title="Tomato Disease Detector",
                       layout="wide")


    def get_base64_of_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    image_path = 'image_background.jpg'
    base64_image = get_base64_of_image(image_path)


    # Custom HTML and CSS for result
    html_content = '''
    <style>
        #result {
            text-align: center;
            margin-top: 20px;
            color: white;
        }
    </style>
    '''

    st.markdown(html_content, unsafe_allow_html=True)

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

    def predict_image(uploaded_file):
        # Make prediction on the uploaded image
        image = np.array(Image.open(uploaded_file))
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = round(float(np.max(predictions[0])) * 100, 2)
        return {
            'predicted_class': predicted_class,
            'confidence': confidence
        }

    st.title("Tomato Disease Detector")
    st.write("Upload an image for prediction")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', width=400)

        # Predict button
        if st.button('Predict'):
            # Make prediction on the uploaded image
            result = predict_image(uploaded_file)
            
            st.markdown(f'''
                <div id="result">
                    <h3>Prediction: {result['predicted_class']}</h3>
                    <p>Confidence: {result['confidence']}%</p>
                </div>
            ''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()