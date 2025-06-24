import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("CIFAR-10 Image Classifier 🎯")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_model.h5")
    return model

model = load_model()

st.write("Choose an image source:")

option = st.radio("Select input method:", ('Upload Image', 'Use Webcam'))

image = None

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
elif option == 'Use Webcam':
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture).convert("RGB")

if image:
    image_resized = image.resize((32, 32))
    st.image(image_resized, caption='Processed Image (resized to 32x32)', use_column_width=False)

    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"### Predicted Class: **{predicted_class}**")
    st.bar_chart(predictions[0])
