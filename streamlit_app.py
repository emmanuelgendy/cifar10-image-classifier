import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model")  # SavedModel folder
    return model


model = load_model()

st.title("CIFAR-10 Image Classifier with Webcam")

# Use Streamlit's built-in webcam input
image_data = st.camera_input("Take a picture")

if image_data is not None:
    # Read image from camera
    image = Image.open(image_data)

    # Preprocess image to match model input
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Prediction: **{predicted_class}**")