import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("CIFAR-10 Image Classifier")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image (32x32)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((32, 32))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    st.write("Prediction:", class_names[np.argmax(prediction)])
